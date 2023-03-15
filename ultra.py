from collections import deque
import paho.mqtt.client as mqtt
import json
import socket
import base64
import threading
import traceback
import random
from GameState import GameState
from StateStaff import StateStaff
from _socket import SHUT_RDWR
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad
from Crypto import Random
import queue
import asyncio
import websockets
import time
import csv
import numpy as np
import pandas as pd
import pywt 
import scipy.signal as sig
from scipy import signal, stats
from scipy.stats import entropy, kurtosis, skew
# from sklearn.feature_selection import SelectKBest
# from sklearn.preprocessing import StandardScaler
# import librosa

# import matplotlib.pyplot as plt
# import pynq
from pynq import Overlay

from ble_packet import BLEPacket
from packet_type import PacketType

from player import Player

"""
Threads: 
1. Client Thread to connect to eval server
2. Server Thread for laptop to connect to
3. Subscriber Thread to publish data to SW Visualizer
4. Game Engine Thread to process incoming data from Laptop

Message Queues:
1. raw_queue between Server and Game Engine Thread
2. eval_queue from Game Engine to Client Thread
3. subscribe_queue from Game Engine to SW Visualizer
"""

raw_queue = deque()
action_queue = deque()
ai_queue = deque()
shot_queue = deque()
subscribe_queue = deque()
fpga_queue = deque()
laptop_queue = deque()
training_model_queue = deque()
eval_queue = deque()
feedback_queue = deque()

collection_flag = False


# Multithread this Chris
class GameEngine(threading.Thread):
    def __init__(self, eval_client):
        super().__init__()

        # queue to receive status from sw
        self.eval_client = eval_client
        self.p1 = self.eval_client.gamestate.player_1
        self.p2 = self.eval_client.gamestate.player_2

        self.shutdown = threading.Event()

    def determine_grenade_hit(self):
        while True:
            print("Random")
            while len(feedback_queue) != 0:
                data = feedback_queue.popleft()
                if data == "6 hit_grenade#":
                    return True
                else:
                    return False

    # one approach is to put it in action queue and continue processing/ or do we want to wait for the grenade actions
    def random_ai_action(self, data):
        actions = ["shoot", "grenade", "shield", "reload", "invalid"]
        action_queue.put(([random.choice(actions)], ["False"]))

    def run(self):

        while not self.shutdown.is_set():
            try:
                if len(action_queue) != 0:
                    action_data, status = action_queue.popleft()

                    print(f"Receive action data by Game Engine: {action_data}")
                    # assuming action_data to be [[p1_action], [p2_status]]

                    if self.p1.shield_status:
                        self.p1.update_shield()

                    if self.p2.shield_status:
                        self.p2.update_shield()

                    if action_data == "logout" or action_data.lower() == 'l':
                        self.p1.action = "logout"
                        # send to visualizer
                        # send to eval server - eval_queue
                        data = self.eval_client.gamestate._get_data_plain_text()
                        subscribe_queue.append(data)
                        # self.eval_client.submit_to_eval()
                        break

                    if action_data == "grenade" or action_data == "G":
                        # receiving the status mqtt topic
                        print("grenade action")
                        if self.p1.throw_grenade():
                            subscribe_queue.append(self.eval_client.gamestate._get_data_plain_text())
                            self.p1.action = "None"
                            # time.sleep(0.5)

                    elif action_data == "shield" or action_data == "S":
                        print("Entered shield action")
                        self.p1.activate_shield()

                    elif action_data == "shoot":
                        print("Entered shoot action")
                        if self.p1.shoot() and status:
                            self.p2.got_shot()

                    elif action_data == "reload" or action_data == "R":
                        self.p1.reload()

                    if action_data == "grenade" or action_data == "G":
                        if self.p1.grenades >= 0:
                            if self.determine_grenade_hit():
                                self.p2.got_hit_grenade()

                                # If health drops to 0 then everything resets except for number of deaths
                    if self.p2.hp <= 0:
                        self.p2.hp = 100
                        self.p2.action = "none"
                        self.p2.bullets = 6
                        self.p2.grenades = 2
                        self.p2.shield_time = 0
                        self.p2.shield_health = 0
                        self.p2.num_shield = 3
                        self.p2.num_deaths += 1

                    # gamestate to eval_server
                    self.eval_client.submit_to_eval()
                    # eval server to subscriber queue
                    self.eval_client.receive_correct_ans()
                    # subscriber queue to sw/feedback queue

                    subscribe_queue.append(self.eval_client.gamestate._get_data_plain_text())

            except KeyboardInterrupt as _:
                traceback.print_exc()

class SubscriberSend(threading.Thread):
    def __init__(self, topic):
        super().__init__()

        # Create a MQTT client
        client = mqtt.Client()
        client.on_message = self.on_message

        self.client = client
        self.topic = topic

        # Flags
        self.shutdown = threading.Event()

    def setup(self):
        print('Setting up connection with HiveMQ')

        self.client.connect("broker.hivemq.com", 1883, 60)
        self.client.subscribe(self.topic)

        print('Successfully connected to HiveMQ and subscribed to topic: ' + self.topic)

    @staticmethod
    def on_message(msg):
        print('Received message: ' + msg.payload.decode())

    def close_connection(self):
        self.client.disconnect()
        self.shutdown.set()

        print("Shutting Down Connection to HiveMQ")

    def send_message(self, message):
        self.client.publish(self.topic, message)

    def run(self):
        self.setup()

        while not self.shutdown.is_set():
            try:
                if len(subscribe_queue) != 0:
                    input_message = subscribe_queue.popleft()

                    print('Publishing to HiveMQ: ', input_message)

                    # if input_message == 'q':
                    #     break
                    self.send_message(input_message)

            except KeyboardInterrupt as _:
                traceback.print_exc()
                self.close_connection()
            except Exception as _:
                traceback.print_exc()
                continue


class SubscriberReceive(threading.Thread):
    def __init__(self, topic):
        super().__init__()

        # Create a MQTT client
        client = mqtt.Client()
        client.connect("broker.hivemq.com")
        client.subscribe(topic)

        client.on_message = self.on_message

        self.client = client
        self.topic = topic

        # Flags
        self.shutdown = threading.Event()

    @staticmethod
    def on_message(client, userdata, message):
        # print("Latency: %.4f seconds" % latency)
        print('Received message: ' + message.payload.decode())
        feedback_queue.append(message.payload.decode())

    def close_connection(self):
        self.client.disconnect()
        self.shutdown.set()

        print("Shutting Down Connection to HiveMQ")

    def run(self):

        while not self.shutdown.is_set():
            try:
                self.client.loop_forever()

            except Exception as e:
                print(e)
                self.close_connection()


class EvalClient:
    def __init__(self, port_num, host_name):
        super().__init__()

        # Create a TCP/IP socket
        self.host_name = host_name
        self.port_num = port_num
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.secret_key = 'chrisisdabest123'
        self.secret_key_bytes = bytes(str(self.secret_key), encoding='utf-8')

        # Create Player
        self.gamestate = GameState()

    def connect_to_eval(self):
        self.connection = self.client_socket.connect((self.host_name, self.port_num))
        print("[EvalClient] connected to eval server")

    def submit_to_eval(self):
        print(f"[EvalClient] Sending plain text gamestate data to the eval server")
        self.gamestate.send_plaintext(self.client_socket)
        print(self.gamestate._get_data_plain_text())

    def receive_correct_ans(self):
        print(f'[EvalClient] Received and update global gamestate')
        self.gamestate.recv_and_update(self.client_socket)
        print(self.gamestate._get_data_plain_text())

    def change(self):
        print('Changing Player stats')
        self.gamestate.init_player(1, 'reload', 50, 3, 1, 0, 0, 2, 1)

    def close_connection(self):
        self.client_socket.close()
        print("Shutting Down EvalClient Connection")

# class EvalClient(threading.Thread):
#     def __init__(self, port_num, host_name):
#         super().__init__()
#
#         # Create a TCP/IP socket
#         client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
#         self.client_socket = client_socket
#         self.connection = client_socket.connect((host_name, port_num))
#         self.secret_key = None
#         self.secret_key_bytes = None
#
#         # Flags
#         self.shutdown = threading.Event()
#
#     def setup(self):
#         print('Defaulting Secret Key to chrisisdabest123')
#
#         # # Blocking Function
#         secret_key = 'chrisisdabest123'
#
#         self.secret_key = secret_key
#         self.secret_key_bytes = bytes(str(self.secret_key), encoding='utf-8')
#
#     def close_connection(self):
#         self.connection.shutdown(SHUT_RDWR)
#         self.connection.close()
#         self.shutdown.set()
#         self.client_socket.close()
#
#         print("Shutting Down EvalClient Connection")
#
#     def decrypt_message(self, message):
#         # Decode from Base64 to Byte Object
#         decode_message = base64.b64decode(message)
#         # Initialization Vector
#         iv = decode_message[:AES.block_size]
#
#         # Create Cipher Object
#         cipher = AES.new(self.secret_key_bytes, AES.MODE_CBC, iv)
#
#         # Obtain Message using Cipher Decrypt
#         decrypted_message_bytes = cipher.decrypt(decode_message[AES.block_size:])
#         # Unpad Message due to AES 16 bytes property
#         decrypted_message_bytes = unpad(decrypted_message_bytes, AES.block_size)
#         # Decode Bytes into utf-8
#         decrypted_message = decrypted_message_bytes.decode("utf-8")
#
#         return decrypted_message
#
#     def encrypt_message(self, message):
#         padded_message = pad(bytes(message, 'utf-8'), AES.block_size)
#
#         iv = Random.new().read(AES.block_size)
#
#         cipher = AES.new(self.secret_key_bytes, AES.MODE_CBC, iv)
#         encrypted_message = iv + cipher.encrypt(padded_message)
#
#         encoded_message = base64.b64encode(encrypted_message).decode('utf-8')
#         return encoded_message
#
#     def run(self):
#         self.setup()
#
#         while not self.shutdown.is_set():
#             try:
#                 if eval_queue:
#                     input_message = eval_queue.popleft()
#
#                     print("Sending Message to Eval Client:", input_message)
#
#                     # if input_message == 'q':
#                     #     break
#
#                     encrypted_message = self.encrypt_message(input_message)
#
#                     # Format String for Eval Server Byte Sequence Process
#                     final_message = str(len(encrypted_message)) + "_" + encrypted_message
#
#                     self.client_socket.sendall(final_message.encode())
#
#                     message_length = self.client_socket.recv(64)
#                     message = self.client_socket.recv(512)
#
#                     correct_status = json.loads(message[0].decode()) # Dictionary Value
#                     action_queue.append(['update', correct_status])
#
#                     laptop_queue.append(correct_status) # Server thread queue to send back to relay laptop
#                     subscribe_queue.append(correct_status) # Subscribe thread queue to update visualizer
#
#                     print('Append to Laptop Queue')
#             except KeyboardInterrupt as _:
#                 traceback.print_exc()
#                 self.close_connection()
#             except Exception as _:
#                 traceback.print_exc()
#                 continue


# class WebSocketServer:
#     def __init__(self, host_name, port_num):
#         self.host_name = host_name
#         self.port_num = port_num
#
#         self.packer = BLEPacket()
#
#         self.data = b''
#
#     async def process_message(self, websocket, path):
#         async for message in websocket:
#             # self.data = self.data + message
#             # if len(self.data) < 20:
#             #     continue
#             # packet = self.data[:20]
#             # self.data = self.data[20:]
#             #
#             # self.packer.unpack(packet)
#             # print("CRC: ", self.packer.get_crc())
#
#             # # await websocket.send(message)
#
#             #
#             # if collection_flag:
#             #     training_model_queue.append(packet)
#
#     async def start_server(self):
#         async with websockets.serve(self.process_message, self.host_name, self.port_num):
#             await asyncio.Future()
#
#     def run(self):
#         asyncio.run(self.start_server())


# class Processing(threading.Thread):
#     shot_flag = False
#     def __init__(self):
#         super().__init__()
#
#     def


class Server(threading.Thread):
    shot_flag = False

    def __init__(self, port_num, host_name):
        super().__init__()

        # Create a TCP/IP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Place Socket into TIME WAIT state
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Binds socket to specified host and port
        server_socket.bind((host_name, port_num))

        # Data Buffer
        self.data = b''

        self.server_socket = server_socket

        # Flags
        self.shutdown = threading.Event()

        self.packer = BLEPacket()

    def setup(self):
        print('Awaiting Connection from Laptop')

        # Blocking Function
        self.connection, client_address = self.server_socket.accept()

        print('Successfully connected to', client_address[0])

    def check_shot(self):
        start_time = time.time()
        while time.time() - start_time < 0.7:
            Server.shot_flag = True
        Server.shot_flag = False

    def close_connection(self):
        self.connection.shutdown(SHUT_RDWR)
        self.connection.close()
        self.shutdown.set()

        print("Shutting Down Server")

    def run(self):
        self.server_socket.listen(1)
        self.setup()

        while not self.shutdown.is_set():
            try:
                # Receive up to 64 Bytes of data
                data = self.connection.recv(64)

                # Append existing data into new data
                self.data = self.data + data

                if len(self.data) < 20:
                    continue

                packet = self.data[:20]
                self.data = self.data[20:]
                self.packer.unpack(packet)

                if self.packer.get_beetle_id() == 1:
                    action_queue.append(["shoot", True])
                    continue
                elif self.packer.get_beetle_id() == 2:
                    print("Someone has been shot")
                    # action_queue.append(["shoot", Server.shot_flag])
                    # continue
                elif self.packer.get_beetle_id() == 3:
                    packet = self.packer.get_euler_data() + self.packer.get_acc_data()
                    ai_queue.append(packet)
                    continue
                else:
                    print('Unknown Beetle ID')

                # # Remove when Training is complete
                # if global_flag:
                #     fpga_queue.put(packet)

                # Sends data back into the relay laptop
                if len(laptop_queue) != 0 :
                    game_state = laptop_queue.popleft()
                    node_id = 0
                    packet_type = PacketType.ACK
                    header = (node_id << 4) | packet_type
                    data = [header, game_state['p1']['bullets'], game_state['p1']['hp'], 0, 0, 0, 0, 0, 0, 0]
                    data = self.packer.pack(data)

                    self.connection.send(data)

                    print("Sending back to laptop", data)
            except KeyboardInterrupt as _:
                traceback.print_exc()
                self.close_connection()
            except Exception as _:
                traceback.print_exc()
                continue

class TrainingModel(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.shutdown = threading.Event()

        self.packer = BLEPacket()

        self.columns = ['flex1', 'flex2', 'gx', 'gy', 'gz', 'accX', 'accY', 'accZ']

        self.factors = ['mean', 'std', 'variance', 'min', 'max', 'range', 'peak_to_peak_amplitude',
                        'mad', 'root_mean_square', 'interquartile_range', 'percentile_75',
                        'skewness', 'kurtosis', 'zero_crossing_rate', 'energy']

        self.headers = [f'{raw_header}_{factor}' for raw_header in self.columns for factor in self.factors]
        self.headers.extend(['action', 'timestamp'])

        self.filename = "/home/kenneth/Desktop/CG4002/training/raw_data.csv"

        # defining game action dictionary
        self.action_map = {0: 'GRENADE', 1: 'LOGOUT', 2: 'SHIELD', 3: 'RELOAD'}

    def sleep(self, seconds):
        start_time = time.time()
        while time.time() - start_time < seconds:
            pass

    def generate_simulated_data(self):
        yaw = random.uniform(-180, 180)
        pitch = random.uniform(-180, 180)
        roll = random.uniform(-180, 180)
        accX = random.uniform(-9000, 9000)
        accY = random.uniform(-9000, 9000)
        accZ = random.uniform(-9000, 9000)
        flex1 = random.uniform(-1, 1)
        flex2 = random.uniform(-1, 1)
        return [flex1, flex2, yaw, pitch, roll, accX, accY, accZ]

        # simulate game movement with noise and action

    def generate_simulated_wave(self):

        # base noise 10s long -> 20Hz*10 = 200 samples
        t = np.linspace(0, 5, 200)  # Define the time range
        x1 = 0.2 * np.sin(t) + 0.2 * np.random.randn(200)
        x1[(x1 > -1) & (x1 < 1)] = 0.0  # TODO - sensor noise within margin of error auto remove

        # movement motion
        period = 2  # seconds
        amplitude = 5
        t = np.linspace(0, 2, int(2 / 0.05))  # Define the time range
        x2 = amplitude * np.sin(2 * np.pi * t / period)[:40]  # Compute the sine wave for only one cycle

        x = x1
        # Add to the 40th-80th elements
        x[20:60] += x2
        x[80:120] += x2

        return x

    def preprocess_data(self, data):
        mean = np.mean(data)
        std = np.std(data)
        variance = np.var(data)
        min = np.min(data)
        max = np.max(data)
        range = np.max(data) - np.min(data)
        peak_to_peak_amplitude = np.abs(np.max(data) - np.min(data))
        mad = np.median(np.abs(data - np.median(data)))
        root_mean_square = np.sqrt(np.mean(np.square(data)))
        interquartile_range = stats.iqr(data)
        percentile_75 = np.percentile(data, 75)
        skewness = stats.skew(data.reshape(-1, 1))[0]
        kurtosis = stats.kurtosis(data.reshape(-1, 1))[0]
        zero_crossing_rate = ((data[:-1] * data[1:]) < 0).sum()
        energy = np.sum(data ** 2)
        # entropy = stats.entropy(data, base=2)

        output_array = [mean, std, variance, min, max, range, peak_to_peak_amplitude,
                        mad, root_mean_square, interquartile_range, percentile_75,
                        skewness, kurtosis, zero_crossing_rate, energy]

        output_array = np.array(output_array)
        return output_array.reshape(1, -1)

    def preprocess_dataset(self, df):
        processed_data = []

        # Loop through each column and compute features
        for column in df.columns:
            column_data = df[column].values
            column_data = column_data.reshape(1, -1)
            print(f"column_data: {column_data}\n")
            print("Data type of column_data:", type(column_data))
            print("Size of column_data:", column_data.size)

            temp_processed = self.preprocess_data(column_data)

            # print(processed_column_data)
            # Append processed column data to main processed data array
            processed_data.append(temp_processed)

        processed_data_arr = np.concatenate(processed_data)

        # reshape into a temporary dataframe of 8x14
        temp_df = pd.DataFrame(processed_data_arr.reshape(8, -1), index=self.columns, columns=self.factors)

        # print the temporary dataframe
        print(f"processed_data: \n {temp_df} \n")
        print(f"len processed_data: {len(processed_data_arr)}\n")

        return processed_data_arr

    def MLP(self, data):
        start_time = time.time()
        # allocate in and out buffer
        in_buffer = pynq.allocate(shape=(24,), dtype=np.double)

        # print time taken so far
        print(f"MLP time taken so far in_buffer: {time.time() - start_time}")
        # out buffer of 1 integer
        out_buffer = pynq.allocate(shape=(1,), dtype=np.int32)
        print(f"MLP time taken so far out_buffer: {time.time() - start_time}")

        # # TODO - copy all data to in buffer
        # for i, val in enumerate(data):
        #     in_buffer[i] = val

        for i, val in enumerate(data[:24]):
            in_buffer[i] = val

        print(f"MLP time taken so far begin trf: {time.time() - start_time}")

        self.dma.sendchannel.transfer(in_buffer)
        self.dma.recvchannel.transfer(out_buffer)

        print(f"MLP time taken so far end trf: {time.time() - start_time}")

        # wait for transfer to finish
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        print(f"MLP time taken so far wait: {time.time() - start_time}")

        # print("mlp done \n")

        # print output buffer
        for output in out_buffer:
            print(f"mlp done with output {output}")

        print(f"MLP time taken so far output: {time.time() - start_time}")

        return [random.random() for _ in range(4)]

    def instantMLP(self, data):
        # Define the input weights and biases
        w1 = np.random.rand(24, 10)
        b1 = np.random.rand(10)
        w2 = np.random.rand(10, 20)
        b2 = np.random.rand(20)
        w3 = np.random.rand(20, 4)
        b3 = np.random.rand(4)

        # Perform the forward propagation
        a1 = np.dot(data[:24], w1) + b1
        h1 = np.maximum(0, a1)  # ReLU activation
        a2 = np.dot(h1, w2) + b2
        h2 = np.maximum(0, a2)  # ReLU activation
        a3 = np.dot(h2, w3) + b3

        c = np.max(a3)
        exp_a3 = np.exp(a3 - c)
        softmax_output = exp_a3 / np.sum(exp_a3)  # Softmax activation

        return softmax_output

    def run(self):
        # Write Row
        # with open("/home/xilinx/code/training/processed_data.csv", "a") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(self.headers)

        while not self.shutdown.is_set():
            try:
                input("Start Training Model?")

                start_time = time.time()

                raw_data = []

                global collection_flag
                collection_flag = True

                training_model_queue.clear()

                while time.time() - start_time < 2:
                    data = training_model_queue.popleft()
                    if len(data) == 8:
                        raw_data.append(data)

                for i in raw_data:
                    self.packer.unpack(i)
                    data = self.packer.get_euler_data() + self.packer.get_acc_data()
                    print(f"data: {data} \n")

                collection_flag = False

                ui = input("Save this Data? y/n")

                if ui.lower() == "y":
                    with open(self.filename, "a") as f:
                        writer = csv.writer(f)
                        for row in raw_data:
                            writer.writerow(row)

                    # Prompt user for label
                    # label = input("Enter label (G = GRENADE, R = RELOAD, S = SHIELD, L = LOGOUT): ")

                    print("Data processed and saved to CSV file.")
                else:
                    raw_data = []
                    i = 0
                    print("not proceed, restart")
            except Exception as _:
                traceback.print_exc()
                continue


class AI(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.shutdown = threading.Event()

        self.columns = ['gx', 'gy', 'gz', 'accX', 'accY', 'accZ']

        self.factors = ['mean', 'std', 'variance', 'range', 'peak_to_peak_amplitude',
                        'mad', 'root_mean_square', 'interquartile_range', 'percentile_75',
                        'energy']

        self.num_groups = 8
        self.headers = [f'grp_{i + 1}_{column}_{factor}' for i in range(self.num_groups)
                        for column in self.columns for factor in self.factors]

        self.headers.extend(['action'])

        # defining game action dictionary
        self.action_map = {0: 'GRENADE', 1: 'LOGOUT', 2: 'SHIELD', 3: 'RELOAD'}

        # PYNQ overlay - TODO
        # self.overlay = Overlay("design_3.bit")
        # self.dma = self.overlay.axi_dma_0

    def sleep(self, seconds):
        start_time = time.time()
        while time.time() - start_time < seconds:
            pass

    def generate_simulated_data(self):
        gx = random.uniform(-180, 180)
        gy = random.uniform(-180, 180)
        gz = random.uniform(-180, 180)
        accX = random.uniform(-9000, 9000)
        accY = random.uniform(-9000, 9000)
        accZ = random.uniform(-9000, 9000)
        return [gx, gy, gz, accX, accY, accZ]

    # simulate game movement with noise and action
    def generate_simulated_wave(self):

        # base noise 10s long -> 20Hz*10 = 200 samples
        t = np.linspace(0, 5, 200)  # Define the time range
        x1 = 0.2 * np.sin(t) + 0.2 * np.random.randn(200)
        x1[(x1 > -1) & (x1 < 1)] = 0.0  # TODO - sensor noise within margin of error auto remove

        # movement motion
        period = 2  # seconds
        amplitude = 5
        t = np.linspace(0, 2, int(2 / 0.05))  # Define the time range
        x2 = amplitude * np.sin(2 * np.pi * t / period)[:40]  # Compute the sine wave for only one cycle

        x = x1
        # Add to the 40th-80th elements
        x[20:60] += x2
        x[80:120] += x2

        return x

    # 10 features
    def preprocess_data(self, data):
        # standard data processing techniques
        mean = np.mean(data)
        std = np.std(data)
        variance = np.var(data)
        range = np.max(data) - np.min(data)
        peak_to_peak_amplitude = np.abs(np.max(data) - np.min(data))
        mad = np.median(np.abs(data - np.median(data)))
        root_mean_square = np.sqrt(np.mean(np.square(data)))
        interquartile_range = stats.iqr(data)
        percentile_75 = np.percentile(data, 75)
        # skewness = stats.skew(data.reshape(-1, 1))[0]
        # kurtosis = stats.kurtosis(data.reshape(-1, 1))[0]
        energy = np.sum(data ** 2)
        output_array = [mean, std, variance, range, peak_to_peak_amplitude,
                        mad, root_mean_square, interquartile_range, percentile_75,
                        energy]

        output_array = np.array(output_array)

        return output_array.reshape(1, -1)

    def preprocess_dataset(self, df):
        processed_data = []

        # Set the window size for the median filter
        window_size = 7

        # Apply the median filter to each column of the DataFrame
        df_filtered = df.rolling(window_size, min_periods=1, center=True).median()

        df = df_filtered

        # Split the rows into 8 groups
        group_size = 5
        data_groups = [df.iloc[i:i + group_size, :] for i in range(0, len(df), group_size)]

        # Loop through each group and column, and compute features
        for group in data_groups:
            group_data = []
            for column in df.columns:
                column_data = group[column].values
                column_data = column_data.reshape(1, -1)

                temp_processed = self.preprocess_data(column_data)
                temp_processed = temp_processed.flatten()

                group_data.append(temp_processed)

            processed_data.append(np.concatenate(group_data))

        # Combine the processed data for each group into a single array
        processed_data_arr = np.concatenate(processed_data)

        print(f"len processed_data_arr={len(processed_data_arr)}\n")

        return processed_data_arr

    def MLP(self, data):
        start_time = time.time()
        # allocate in and out buffer
        in_buffer = pynq.allocate(shape=(24,), dtype=np.double)

        # print time taken so far
        print(f"MLP time taken so far in_buffer: {time.time() - start_time}")
        # out buffer of 1 integer
        out_buffer = pynq.allocate(shape=(1,), dtype=np.int32)
        print(f"MLP time taken so far out_buffer: {time.time() - start_time}")

        # # TODO - copy all data to in buffer
        # for i, val in enumerate(data):
        #     in_buffer[i] = val

        for i, val in enumerate(data[:24]):
            in_buffer[i] = val

        print(f"MLP time taken so far begin trf: {time.time() - start_time}")

        self.dma.sendchannel.transfer(in_buffer)
        self.dma.recvchannel.transfer(out_buffer)

        print(f"MLP time taken so far end trf: {time.time() - start_time}")

        # wait for transfer to finish
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        print(f"MLP time taken so far wait: {time.time() - start_time}")

        # print("mlp done \n")

        # print output buffer
        for output in out_buffer:
            print(f"mlp done with output {output}")

        print(f"MLP time taken so far output: {time.time() - start_time}")

        return [random.random() for _ in range(4)]

    def instantMLP(self, data):
        # Define the input weights and biases
        # w1 = np.random.rand(24, 10)
        # b1 = np.random.rand(10)
        # w2 = np.random.rand(10, 20)
        # b2 = np.random.rand(20)
        # w3 = np.random.rand(20, 4)
        # b3 = np.random.rand(4)

        # # Perform the forward propagation
        # a1 = np.dot(data[:24], w1) + b1
        # h1 = np.maximum(0, a1)  # ReLU activation
        # a2 = np.dot(h1, w2) + b2
        # h2 = np.maximum(0, a2)  # ReLU activation
        # a3 = np.dot(h2, w3) + b3

        # c = np.max(a3)
        # exp_a3 = np.exp(a3 - c)
        # softmax_output = exp_a3 / np.sum(exp_a3)  # Softmax activation

        # return softmax_output

        # Load the model from file and preproessing
        # localhost
        mlp = joblib.load('mlp_model.joblib')
        scaler = joblib.load('scaler.joblib')
        pca = joblib.load('pca.joblib')

        # board
        # mlp = joblib.load('/home/xilinx/mlp_model.joblib')
        # scaler = joblib.load('/home/xilinx/scaler.joblib')
        # pca = joblib.load('/home/xilinx/pca.joblib')

        test_data_std = scaler.transform(data.reshape(1, -1))
        test_data_pca = pca.transform(test_data_std)

        # Use the loaded MLP model to predict labels for the test data
        predicted_labels = mlp.predict(test_data_pca)

        predicted_label = str(predicted_labels[0].item())  # convert to single char

        return predicted_label

    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):

        # live integration loop
        # while not self.shutdown.is_set():
        f = True
        while f:
            f = False

            df = pd.DataFrame(columns=self.columns)
            # Define the window size and threshold factor
            window_size = 11
            threshold_factor = 2

            # Define N units for flagging movement, 20Hz -> 2s = 40 samples
            N = 40

            # Initialize empty arrays for data storage
            t = []
            x = []
            filtered = []
            threshold = []
            movement_detected = []
            last_movement_time = -N  # set last movement time to negative N seconds ago
            wave = self.generate_simulated_wave()
            i = 0
            timenow = 0

            print(f"entering while loop \n")

            while True:
                # Create plot window
                # plt.ion()
                # plt.show()

                data = ai_queue.popleft()
                # self.sleep(0.05)
                print("Data: ")
                print(" ".join([f"{x:.8g}" for x in data]))
                print("\n")

                # Append new data to dataframe
                df.loc[len(df)] = data

                # Compute absolute acceleration values
                # x.append(np.abs(data[5:8])) # abs of accX, accY, accZ
                x.append(wave[i])  # abs of accX, accY, accZ

                # time
                t.append(timenow)

                # Compute moving window median
                if len(x) < window_size:
                    filtered.append(0)
                else:
                    filtered.append(np.median(x[-window_size:], axis=0))

                # Compute threshold using past median data, threshold = mean + k * std
                if len(filtered) < window_size:
                    threshold.append(0)
                else:
                    past_filtered = filtered[-window_size:]
                    threshold.append(
                        np.mean(past_filtered, axis=0) + (threshold_factor * np.std(past_filtered, axis=0)))

                # Identify movement
                if len(filtered) > window_size:
                    # checking if val is past threshold and if last movement was more than N samples ago
                    if np.all(filtered[-1] > threshold[-1]) and len(t) - last_movement_time >= N:
                        movement_detected.append(len(df) - 1)
                        last_movement_time = len(t)  # update last movement time
                        print(f"Movement detected at sample {len(df) - 1}")

                # if movement has been detected for more than N samples, preprocess and feed into neural network
                if len(movement_detected) > 0 and len(df) - movement_detected[-1] >= N:
                    # extract movement data
                    start = movement_detected[-1]
                    end = len(df)
                    movement_data = df.iloc[start:end, :]

                    # print the start and end index of the movement
                    print(f"Processing movement detected from sample {start} to {end}")

                    # perform data preprocessing
                    preprocessed_data = self.preprocess_dataset(movement_data)

                    # print preprocessed data
                    print(f"preprocessed data to feed into MLP: \n {preprocessed_data} \n")

                    # feed preprocessed data into neural network
                    # output = self.MLP(preprocessed_data)
                    predicted_label = self.instantMLP(preprocessed_data)

                    action_queue.append([predicted_label, False]) # G, L, R, S

                    print(f"output from MLP: \n {predicted_label} \n")  # print output of MLP

                    # np_output = np.array(output)
                    # largest_index = np_output.argmax()

                    # largest_action = self.action_map[largest_index]

                    # print largest index and largest action of MLP output
                    # print(f"largest index: {largest_index} \n")
                    # print(f"largest action: {largest_action} \n")

                    # reset movement_detected list
                    movement_detected.clear()

                i += 1
                timenow += 1

                if i == 200:
                    i = 0

            # except Exception as _:
            #     traceback.print_exc()
            #     self.close_connection()
            #     print("an error occurred")


if __name__ == '__main__':
    print('---------------<Announcement>---------------')

    # Software Visualizer
    # print("Starting Subscriber Thread        ")
    hive = SubscriberSend("CG4002")
    hive.start()

    # Starting Visualizer Receive
    viz = SubscriberReceive("gamestate")
    viz.start()

    # Client Connection to Evaluation Server
    print("Starting Client Thread           ")
    eval_client = EvalClient(1234, "localhost")
    eval_client.connect_to_eval()
    # input("block")
    # eval_client.submit_to_eval()
    # eval_client.receive_correct_ans()
    # eval_client.change()
    # eval_client.submit_to_eval()
    # eval_client.receive_correct_ans()

    # Game Engine
    print("Starting Game Engine Thread        ")
    GE = GameEngine(eval_client=eval_client)
    GE.start()

    # AI Model
    #print("Starting AI Model Thread")
    #ai_model = Training()
    #ai_model.start()

    # Server Connection to Laptop
    # print("Starting Server Thread           ")
    # laptop_server = Server(8080, "192.168.95.221")
    # laptop_server.start()

    # print("Starting Web Socket Server Thread")
    # laptop_server = WebSocketServer("192.168.95.221", 8080)
    # laptop_server.run()

    print('--------------------------------------------')

    while True:
        action = input("Action: ")
        action_queue.append([action, True])
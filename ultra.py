import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import websockets
import threading
import traceback
import constants
import asyncio
import socket
import random
import joblib
import time
import json
import csv
import pynq
import librosa
from pynq import Overlay
from GameState import GameState
from _socket import SHUT_RDWR
from scipy import stats
from queue import Queue
from collections import deque
from ble_packet import BLEPacket
from packet_type import PacketType
# from sklearn.feature_selection import SelectKBest
# from sklearn.preprocessing import StandardScaler

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
subscribe_queue = Queue()
fpga_queue = deque()
laptop_queue = deque()
training_model_queue = deque()
eval_queue = deque()
feedback_queue = Queue()

collection_flag = False


class ShootEngine(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.gun_shot = False
        self.vest_shot = False

    def handle_gun_shot(self):
        self.gun_shot = True

    def handle_vest_shot(self):
        self.vest_shot = True

    def run(self):
        while True:
            if self.gun_shot:
                self.gun_shot = False
                time.sleep(0.5)
                if self.vest_shot:
                    action_queue.append(['shoot', True])
                    self.vest_shot = False
                else:
                    action_queue.append(['shoot', False])
            if self.vest_shot:
                self.vest_shot = False
                time.sleep(0.5)
                if self.gun_shot:
                    action_queue.append(['shoot', True])
                    self.gun_shot = False
                else:
                    action_queue.append(['shoot', False])


class GameEngine(threading.Thread):
    def __init__(self, eval_client):
        super().__init__()

        # queue to receive status from sw
        self.eval_client = eval_client
        self.p1 = self.eval_client.gamestate.player_1
        self.p2 = self.eval_client.gamestate.player_2

        self.shutdown = threading.Event()

    def determine_grenade_hit(self):
        '''
        while True:
            print("Random")
            while not feedback_queue.empty():
                data = feedback_queue.get()
                if data == "6 hit_grenade#":
                    return True
                else:
                    return False
        '''
        return True
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
                        subscribe_queue.put(data)
                        # self.eval_client.submit_to_eval()
                        break

                    if action_data == "grenade" or action_data == "G":
                        # receiving the status mqtt topic
                        print("grenade action")
                        if self.p1.throw_grenade():
                            subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())
                        
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

                    subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())

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
                if not subscribe_queue.empty():
                    input_message = subscribe_queue.get()

                    print('Publishing to HiveMQ: ', input_message)

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
        # print('Received message: ' + message.payload.decode())
        feedback_queue.put(message.payload.decode())

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

        self.secret_key = 'ilovecg4002profs'
        self.secret_key_bytes = bytes(str(self.secret_key), encoding='utf-8')

        # Create Player
        self.gamestate = GameState()

    def connect_to_eval(self):
        self.connection = self.client_socket.connect((self.host_name, self.port_num))
        print("[EvalClient] connected to eval server")

    def submit_to_eval(self):
        print(f'[EvalClient] Sent Game State: {self.gamestate._get_data_plain_text()}'.ljust(80), end='\r')
        self.gamestate.send_plaintext(self.client_socket)

    def receive_correct_ans(self):
        print(f'[EvalClient] Received Game State: {self.gamestate._get_data_plain_text()}'.ljust(80), end='\r')
        self.gamestate.recv_and_update(self.client_socket)

    def close_connection(self):
        self.client_socket.close()
        print("Shutting Down EvalClient Connection")


class Server(threading.Thread):
    def __init__(self, port_num, host_name):
        super().__init__()

        # Create a TCP/IP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Place Socket into TIME WAIT state
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Binds socket to specified host and port
        server_socket.bind((host_name, port_num))

        self.server_socket = server_socket

        self.packer = BLEPacket()

        self.shoot_engine = ShootEngine()

        # Data Buffer
        self.data = b''

        # Flags
        self.shutdown = threading.Event()

    def setup(self):
        print('Awaiting Connection from Laptop')

        # Blocking Function
        self.connection, client_address = self.server_socket.accept()

        print('Successfully connected to', client_address[0])

    def close_connection(self):
        self.connection.shutdown(SHUT_RDWR)
        self.connection.close()
        self.shutdown.set()

        print("Shutting Down Server")

    def run(self):
        shot_thread = threading.Thread(target=self.shoot_engine.start)
        shot_thread.start()
        self.server_socket.listen(1)
        self.setup()

        while not self.shutdown.is_set():
            try:
                # Receive up to 64 Bytes of data
                data = self.connection.recv(64)
                # Append existing data into new data
                self.data = self.data + data

                if len(self.data) < constants.packet_size:
                    continue

                packet = self.data[:constants.packet_size]
                self.data = self.data[constants.packet_size:]
                self.packer.unpack(packet)

                packet_id = self.packer.get_beetle_id()

                if packet_id == 1:
                    self.shoot_engine.handle_gun_shot()
                elif packet_id == 2:
                    self.shoot_engine.handle_vest_shot()
                elif packet_id == 3:
                    packet = self.packer.get_euler_data() + self.packer.get_acc_data()
                    ai_queue.append(packet)
                else:
                    print("Invalid Beetle ID")

                # # Remove when Training is complete
                # if global_flag:
                #     fpga_queue.put(packet)

                # Sends data back into the relay laptop
                if len(laptop_queue) != 0:
                    game_state = laptop_queue.popleft()
                    data = [0, game_state['p1']['bullets'], game_state['p1']['hp'], 0, game_state['p2']['bullets'],
                            game_state['p2']['hp'], 0]
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


class AIModel(threading.Thread):
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
        self.action_map = {0: 'G', 1: 'L', 2: 'R', 3: 'S'}

        # load PCA model
        # read the contents of the arrays.txt file
        with open("arrays.txt", "r") as f:
            data = json.load(f)

        # extract the weights and bias arrays
        self.scaling_factor = data['scaling_factor']
        self.mean = data['mean']
        self.variance = data['variance']
        self.pca_eigvecs_list = data['pca_eigvecs_list']

        self.pca_eigvecs_transposed = [list(row) for row in zip(*self.pca_eigvecs_list)]
        # PYNQ overlay
        # self.overlay = Overlay("pca_mlp_1.bit")
        # self.dma = self.overlay.axi_dma_0

        # # Allocate input and output buffers once
        # self.in_buffer = pynq.allocate(shape=(35,), dtype=np.float32)
        # self.out_buffer = pynq.allocate(shape=(4,), dtype=np.float32)

    def sleep(self, seconds):
        start_time = time.time()
        while time.time() - start_time < seconds:
            pass

    def generate_simulated_data(self):
        gx = random.uniform(-9, 9)  # TODO - assumption: gyro x,y,z change btwn -9 to 9
        gy = random.uniform(-9, 9)
        gz = random.uniform(-9, 9)
        accX = random.uniform(-9, 9)
        accY = random.uniform(-9, 9)
        accZ = random.uniform(-9, 9)
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
        energy = np.sum(data ** 2)

        output_array = np.empty((1, 10))
        output_array[0] = [mean, std, variance, range, peak_to_peak_amplitude, mad, root_mean_square,
                           interquartile_range, percentile_75, energy]

        return output_array

    def preprocess_dataset(self, arr):
        processed_data = []

        # Set the window size for the median filter
        window_size = 7

        df = pd.DataFrame(arr)
        df_filtered = df.rolling(window_size, min_periods=1, center=True).median()

        arr = df_filtered.values

        # Split the rows into 8 groups
        group_size = 5
        num_groups = 8

        # Loop through each group and column, and compute features
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            group = arr[start_idx:end_idx, :]

            group_data = []
            for column in range(arr.shape[1]):
                column_data = group[:, column]
                column_data = column_data.reshape(1, -1)

                temp_processed = self.preprocess_data(column_data)
                temp_processed = temp_processed.flatten()

                group_data.append(temp_processed)

            processed_data.append(np.concatenate(group_data))

        processed_data_arr = np.concatenate(processed_data)

        print(f"len processed_data_arr={len(processed_data_arr)}\n")

        return processed_data_arr

    def PCA_MLP(self, data):
        start_time = time.time()

        # reshape data to match in_buffer shape
        data = np.reshape(data, (35,))

        self.in_buffer[:] = data

        self.dma.sendchannel.transfer(self.in_buffer)
        self.dma.recvchannel.transfer(self.out_buffer)

        # wait for transfer to finish
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # print output buffer
        print("mlp done with output: " + " ".join(str(x) for x in self.out_buffer))

        print(f"MLP time taken so far output: {time.time() - start_time}")

        return self.out_buffer

    def instantMLP(self, data):
        # Load the model from file and preproessing
        # localhost
        mlp = joblib.load('mlp_model.joblib')

        # board
        # mlp = joblib.load('/home/xilinx/mlp_model.joblib')

        # sample data for sanity check
        test_input = np.array([0.1, 0.2, 0.3, 0.4] * 120).reshape(1, -1)

        # Scaler
        # test_input_rescaled = (data - self.mean) / np.sqrt(self.variance) # TODO - use this for real data
        test_input_rescaled = (test_input - self.mean) / np.sqrt(self.variance)
        print(f"test_input_rescaled: {test_input_rescaled}\n")

        # PCA
        test_input_math_pca = np.dot(test_input_rescaled, self.pca_eigvecs_transposed)
        print(f"test_input_math_pca: {test_input_math_pca}\n")

        arr = np.array([-9.20434773, -4.93421279, -0.7165668, -5.35652778, 1.16597442, 0.83953718,
                        2.46925983, 0.55131264, -0.1671036, 0.82080829, -1.87265269, 3.34199444,
                        0.09530707, -3.77394007, 1.68183889, 1.97630386, 1.48839111, -3.00986825,
                        4.13786954, 1.46723819, 8.08842927, 10.94846901, 2.22280215, -1.85681443,
                        4.47327707, 3.15918201, -0.77879694, -0.11557772, 0.21580221, -2.62405631,
                        -3.42924226, -7.01213438, 7.75544419, -3.72408571, 3.46613566])

        assert np.allclose(test_input_math_pca, arr)

        # MLP
        # predicted_labels = self.PCA_MLP(test_input_math_pca) # return 1x4 softmax array
        # print(f"MLP pynq overlay predicted: {predicted_labels} \n")
        # np_output = np.array(predicted_labels)
        # largest_index = np_output.argmax()

        # predicted_label = self.action_map[largest_index]

        # # print largest index and largest action of MLP output
        # print(f"largest index: {largest_index} \n")
        # print(f"MLP overlay predicted: {predicted_label} \n")

        predicted_label = mlp.predict(test_input_math_pca.reshape(1, -1))
        print(f"MLP lib overlay predicted: {predicted_label} \n")

        # output is a single char
        return predicted_label

    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        # live integration loop
        window_size = 11
        threshold_factor = 2

        buffer_size = 500
        buffer = np.zeros((buffer_size, len(self.columns)))
        # Define the window size and threshold factor

        # Define N units for flagging movement, 20Hz -> 2s = 40 samples
        N = 40

        # Initialize empty arrays for data storage
        x = np.zeros(buffer_size)
        filtered = np.zeros(buffer_size)
        threshold = np.zeros(buffer_size)
        last_movement_time = -N  # set last movement time to negative N seconds ago
        # wave = self.generate_simulated_wave()
        i = 0
        buffer_index = 0

        # while not self.shutdown.is_set():
        while True:
            if ai_queue:
                data = ai_queue.popleft()
                # data = self.generate_simulated_data()
                # self.sleep(0.05)
                print("Data: ")
                print(" ".join([f"{x:.8g}" for x in data]))
                print("\n")

                # Append new data
                buffer[buffer_index] = data

                # Update circular buffer index
                buffer_index = (buffer_index + 1) % buffer_size

                # Compute absolute acceleration values
                x[buffer_index] = np.abs(np.sum(np.square(data[3:6])))  # abs of accX, accY, accZ
                # x[buffer_index] = wave[i]  # abs of accX, accY, accZ

                i += 1
                if i >= len(wave):
                    i = 0

                # Compute moving window median
                if buffer_index < window_size:
                    filtered[buffer_index] = 0
                else:
                    filtered[buffer_index] = np.median(x[buffer_index - window_size + 1:buffer_index + 1], axis=0)

                # Compute threshold using past median data, threshold = mean + k * std
                if buffer_index < window_size:
                    threshold[buffer_index] = 0
                else:
                    past_filtered = filtered[buffer_index - window_size + 1:buffer_index + 1]
                    threshold[buffer_index] = np.mean(past_filtered, axis=0) + (
                                threshold_factor * np.std(past_filtered, axis=0))

                # Identify movement
                if buffer_index >= window_size and np.all(
                        filtered[buffer_index] > threshold[buffer_index]) and buffer_index - last_movement_time >= N:
                    last_movement_time = buffer_index  # update last movement time
                    print(f"Movement detected at sample {buffer_index}")

                # if N samples from last movement time have been accumulated, preprocess and feed into neural network
                if (buffer_index - last_movement_time) % buffer_size == N - 1:
                    # extract movement data
                    start = (last_movement_time + 1) % buffer_size
                    end = (buffer_index + 1) % buffer_size
                    if end <= start:
                        movement_data = np.concatenate((buffer[start:, :], buffer[:end, :]), axis=0)
                    else:
                        movement_data = buffer[start:end, :]

                    # print the start and end index of the movement
                    print(f"Processing movement detected from sample {start} to {end}")

                    # perform data preprocessing
                    preprocessed_data = self.preprocess_dataset(movement_data)

                    # feed preprocessed data into neural network
                    predicted_label = self.instantMLP(preprocessed_data)

                    print(f"output from MLP: \n {predicted_label} \n")  # print output of MLP

            # except Exception as _:
            #     traceback.print_exc()
            #     self.close_connection()
            #     print("an error occurred")


class WebSocketServer:
    def __init__(self):
        super().__init__()

    async def receive(self, websocket):
        async for message in websocket:
            print(f'Message Received: {message}'.ljust(40), end='\r')
            await websocket.send(message.encode())

    async def start_server(self):
        async with websockets.serve(self.receive, constants.xilinx_server, constants.xilinx_port_num):
            await asyncio.Future()


if __name__ == '__main__':
    print('---------------<Setup Announcement>---------------')
    # AI Model
    print("Starting AI Model Thread")
    ai_model = AIModel()

    # Software Visualizer
    print("Starting Subscriber Send Thread")
    hive = SubscriberSend("CG4002")

    # Starting Visualizer Receive
    print("Starting Subscribe Receive")
    viz = SubscriberReceive("gamestate")

    # Client Connection to Evaluation Server
    print("Starting Client Thread")
    # eval_client = EvalClient(9999, "137.132.92.184")
    eval_client = EvalClient(constants.eval_port_num, "localhost")
    eval_client.connect_to_eval()

    # Game Engine
    print("Starting Game Engine Thread")
    game_engine = GameEngine(eval_client=eval_client)

    # Server Connection to Laptop
    print("Starting Server Thread")
    laptop_server = Server(constants.xilinx_port_num, constants.xilinx_server)

    # print("Starting Web Socket Server Thread")
    # server = WebSocketServer()
    # asyncio.run(server.start_server())

    print('--------------------------------------------------')

    hive.start()
    viz.start()
    game_engine.start()
    ai_model.start()
    laptop_server.start()






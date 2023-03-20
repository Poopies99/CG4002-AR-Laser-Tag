# import matplotlib.pyplot as plt
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
import csv
# import pynq
# import librosa
# from sklearn.feature_selection import SelectKBest
# from sklearn.preprocessing import StandardScaler
from GameState import GameState
from _socket import SHUT_RDWR
from scipy import stats
from queue import Queue
from collections import deque
from ble_packet import BLEPacket
# from pynq import Overlay

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

        self.secret_key = 'chrisisdabest123'
        self.secret_key_bytes = bytes(str(self.secret_key), encoding='utf-8')

        # Create Player
        self.gamestate = GameState()

    def connect_to_eval(self):
        self.connection = self.client_socket.connect((self.host_name, self.port_num))
        print("[EvalClient] connected to eval server")

    def submit_to_eval(self):
        # print(f"[EvalClient] Sending plain text gamestate data to the eval server")
        self.gamestate.send_plaintext(self.client_socket)
        print(self.gamestate._get_data_plain_text())

    def receive_correct_ans(self):
        # print(f'[EvalClient] Received and update global gamestate')
        self.gamestate.recv_and_update(self.client_socket)
        print(self.gamestate._get_data_plain_text())

    def close_connection(self):
        self.client_socket.close()
        print("Shutting Down EvalClient Connection")


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
        
        start_time = time.time()

        while not self.shutdown.is_set():
            try:
                if time.time() - start_time > 6:
                    action_queue.append(['grenade', True])
                    start_time = time.time()
                    continue
                else:
                    action_queue.append(['shoot', True])
                    '''
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
                        start_time = time.time() 
                    # action_queue.append(["shoot", Server.shot_flag])
                    # continue
                    '''
                '''

                elif self.packer.get_beetle_id() == 3:
                    packet = self.packer.get_euler_data() + self.packer.get_acc_data()
                    ai_queue.append(packet)
                    continue
                else:
                    continue
                '''
                #action_queue.append(["shoot", True])
                #action_queue.append(["grenade", True])

                # # Remove when Training is complete
                # if global_flag:
                #     fpga_queue.put(packet)

                # Sends data back into the relay laptop
                # if len(laptop_queue) != 0 :
                #     game_state = laptop_queue.popleft()
                #     node_id = 0
                #     packet_type = PacketType.ACK
                #     header = (node_id << 4) | packet_type
                #     data = [header, game_state['p1']['bullets'], game_state['p1']['hp'], 0, 0, 0, 0, 0, 0, 0]
                # #     data = [header, game_state['p1']['bullets'], game_state['p1']['hp'], game_state['p2']['bullets'], game_state['p2']['hp'], 0, 0, 0, 0, 0]
                #     data = self.packer.pack(data)
                #
                #     self.connection.send(data)
                #
                #     print("Sending back to laptop", data)
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
        self.action_map = {0: 'G', 1: 'L', 2: 'R', 3: 'S'}

        # PYNQ overlay
        # self.overlay = Overlay("pca_mlp_1.bit")
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

        # print(f"len processed_data_arr={len(processed_data_arr)}\n")

        return processed_data_arr

    def PCA_MLP(self, data):
        start_time = time.time()
        # allocate in and out buffer
        in_buffer = pynq.allocate(shape=(35,), dtype=np.double)  # 1x35 PCA input
        out_buffer = pynq.allocate(shape=(4,), dtype=np.double)  # 1x4 softmax output

        # reshape data to match in_buffer shape
        data = np.reshape(data, (35,))

        for i, val in enumerate(data):
            in_buffer[i] = val

        self.dma.sendchannel.transfer(in_buffer)
        self.dma.recvchannel.transfer(out_buffer)

        # wait for transfer to finish
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # print output buffer
        # print("mlp done with output: " + " ".join(str(x) for x in out_buffer))

        # print(f"MLP time taken so far output: {time.time() - start_time}")

        return out_buffer

    def instantMLP(self, data):
        # Load the model from file and preproessing
        # localhost
        # mlp = joblib.load('mlp_model.joblib')
        # scaler = joblib.load('scaler.joblib')
        # pca = joblib.load('pca.joblib')

        # board
        mlp = joblib.load('/home/xilinx/gunnit_test/mlp_model.joblib')
        scaler = joblib.load('/home/xilinx/gunnit_test/scaler.joblib')
        pca = joblib.load('/home/xilinx/gunnit_test/pca.joblib')
        
        print("enter MLP function")
        # Preprocess data
        test_data_std = scaler.transform(data.reshape(1, -1))
        test_data_pca = pca.transform(test_data_std)

        print("data processed, feeding into mlp")
        # Use MLP
        predicted_labels = mlp.predict(test_data_pca)
        predicted_label = str(predicted_labels[0].item())  # return single char
        
        print("MLP done")
        # print predicted label of MLP predicted_label
        # print(f"MLP lib predicted: {predicted_label} \n")

        # predicted_labels = self.PCA_MLP(test_data_pca)  # return 1x4 softmax array

        # np_output = np.array(predicted_labels)
        # largest_index = np_output.argmax()

        # predicted_label = self.action_map[largest_index]
        # predicted_label = self.action_map[largest_index]

        # print largest index and largest action of MLP output
        # print(f"largest index: {largest_index} \n")
        print(f"MLP overlay predicted: {predicted_label} \n")

        # output is a single char
        return predicted_label

    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        # live integration loop
        # while not self.shutdown.is_set():
        if 1 == 1:
            df = pd.DataFrame(np.zeros((500, len(self.columns))), columns=self.columns)
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
            buffer_index = 0

            print(f"entering while loop \n")

            while True:
                # Create plot window
                # plt.ion()
                # plt.show()
                if len(ai_queue) != 0:
                    data = ai_queue.popleft()
                    # self.sleep(0.05)
                    # print("Data: ")
                    # print(" ".join([f"{x:.8g}" for x in data]))
                    # print("\n")

                    # Append new data to dataframe
                    # Append new data to dataframe
                    df.iloc[buffer_index] = data

                    # Increment buffer index and reset to zero if we reach the end of the buffer
                    buffer_index += 1
                    if buffer_index >= 500:
                        buffer_index = 0

                    # Compute absolute acceleration values
                    x.append(np.abs(data[3:6]))  # abs of accX, accY, accZ
                    # x.append(wave[i])  # abs of accX, accY, accZ

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
                        if np.all(filtered[-1] > threshold[-1]) and len(filtered) - last_movement_time >= N:
                            movement_detected.append(buffer_index)
                            last_movement_time = len(filtered)  # update last movement time
                            print(f"Movement detected at sample {buffer_index}")

                    # if movement has been detected for more than N samples, preprocess and feed into neural network
                    if len(movement_detected) > 0 and buffer_index - movement_detected[-1] >= N:
                        # extract movement data
                        start = movement_detected[-1]
                        end = buffer_index if buffer_index > start else buffer_index + 500
                        movement_data = df.iloc[start:end, :]

                        # print the start and end index of the movement
                        print(f"Processing movement detected from sample {start} to {end}")

                        # perform data preprocessing
                        preprocessed_data = self.preprocess_dataset(movement_data)
                        # feed preprocessed data into neural network
                        # output = self.MLP(preprocessed_data)
                        predicted_label = self.instantMLP(preprocessed_data)

                        print(f"output from MLP: \n {predicted_label} \n")  # print output of MLP

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


class WebSocketServer:
    def __init__(self):
        super().__init__()

    async def echo(self, websocket, path):
        async for message in websocket:
            print(f'Message Received: {message}'.ljust(40))
            await websocket.send(message)

    async def start_server(self):
        async with websockets.serve(self.echo, constants.xilinx_server, constants.xilinx_port_num):
            await asyncio.Future()


if __name__ == '__main__':
    print('---------------<Setup Announcement>---------------')

    # Software Visualizer
    # print("Starting Subscriber Send Thread        ")
    # hive = SubscriberSend("CG4002")
    # hive.start()

    # Starting Visualizer Receive
    # print("Starting Subscribe Receive")
    # viz = SubscriberReceive("gamestate")
    # viz.start()

    # Client Connection to Evaluation Server
    # print("Starting Client Thread           ")
    # eval_client = EvalClient(9999, "137.132.92.184")
    # eval_client = EvalClient(constants.eval_port_num, "localhost")
    # eval_client.connect_to_eval()

    # input("block")
    # eval_client.submit_to_eval()
    # eval_client.receive_correct_ans()
    # eval_client.submit_to_eval()
    # eval_client.receive_correct_ans()

    # Game Engine
    # print("Starting Game Engine Thread        ")
    # GE = GameEngine(eval_client=eval_client)
    # GE.start()

    # AI Model
    #print("Starting AI Model Thread")
    #ai_model = AI()
    #ai_model.start()

    # Server Connection to Laptop
    # print("Starting Server Thread           ")
    # laptop_server = Server(constants.xilinx_port_num, constants.xilinx_server)
    # laptop_server.start()

    print("Starting Web Socket Server Thread")
    server = WebSocketServer()
    asyncio.run(server.start_server())

    # asyncio.run(laptop_server.start_server())

    print('--------------------------------------------------')

import paho.mqtt.client as mqtt
import json
import socket
import base64
import threading
import traceback
import random
from _socket import SHUT_RDWR
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad
from Crypto import Random
from queue import Queue

import time
import csv
import numpy as np
import pandas as pd
import pywt 
import scipy.signal as sig
from scipy import signal, stats
from scipy.stats import entropy, kurtosis, skew
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
import librosa

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

raw_queue = Queue()
eval_queue = Queue()
subscribe_queue = Queue()
fpga_queue = Queue()
laptop_queue = Queue()


class GameEngine(threading.Thread):
    """
    1 Player Game Engine
    """
    def __init__(self):
        super().__init__()

        # Create Player
        self.player = Player()

        # Flags
        self.shutdown = threading.Event()

    def update(self, action):
        try:
            if action == 'shoot':
                return self.player.shoot()
            elif action == 'grenade':
                return self.player.throw_grenade()
            elif action == 'shield':
                return self.player.activate_shield()
            elif action == 'reload':
                return self.player.reload()
            self.player.update_json()
        except Exception as _:
            self.shutdown.set()

    def run(self):
        while not self.shutdown.is_set():
            try:
                input_message = raw_queue.get()

                with open('example.json', 'r') as f:
                    json_data = f.read()

                eval_queue.put(json_data)
                subscribe_queue.put(json_data)

                json_data = json.loads(json_data)
                laptop_queue.put(json_data)
            except Exception as _:
                traceback.print_exc()


class Subscriber(threading.Thread):
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
                input_message = subscribe_queue.get()
                # print('Publishing to HiveMQ: ', input_message)
                if input_message == 'q':
                    break
                self.send_message(input_message)
            except Exception as _:
                traceback.print_exc()
                self.close_connection()


class EvalClient(threading.Thread):
    def __init__(self, port_num, host_name):
        super().__init__()

        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.client_socket = client_socket
        self.connection = client_socket.connect((host_name, port_num))
        self.secret_key = None
        self.secret_key_bytes = None

        # Flags
        self.shutdown = threading.Event()

    def setup(self):
        print('Defaulting Secret Key to chrisisdabest123')

        # # Blocking Function
        secret_key = 'chrisisdabest123'

        self.secret_key = secret_key
        self.secret_key_bytes = bytes(str(self.secret_key), encoding='utf-8')

    def close_connection(self):
        self.connection.shutdown(SHUT_RDWR)
        self.connection.close()
        self.shutdown.set()
        self.client_socket.close()

        print("Shutting Down EvalClient Connection")

    def decrypt_message(self, message):
        # Decode from Base64 to Byte Object
        decode_message = base64.b64decode(message)
        # Initialization Vector
        iv = decode_message[:AES.block_size]

        # Create Cipher Object
        cipher = AES.new(self.secret_key_bytes, AES.MODE_CBC, iv)

        # Obtain Message using Cipher Decrypt
        decrypted_message_bytes = cipher.decrypt(decode_message[AES.block_size:])
        # Unpad Message due to AES 16 bytes property
        decrypted_message_bytes = unpad(decrypted_message_bytes, AES.block_size)
        # Decode Bytes into utf-8
        decrypted_message = decrypted_message_bytes.decode("utf-8")

        return decrypted_message

    def encrypt_message(self, message):
        padded_message = pad(bytes(message, 'utf-8'), AES.block_size)

        iv = Random.new().read(AES.block_size)

        cipher = AES.new(self.secret_key_bytes, AES.MODE_CBC, iv)
        encrypted_message = iv + cipher.encrypt(padded_message)

        encoded_message = base64.b64encode(encrypted_message).decode('utf-8')
        return encoded_message

    def run(self):
        self.setup()

        while not self.shutdown.is_set():
            try:
                input_message = eval_queue.get()
                if input_message == 'q':
                    break

                encrypted_message = self.encrypt_message(input_message)

                # Format String for Eval Server Byte Sequence Process
                final_message = str(len(encrypted_message)) + "_" + encrypted_message

                self.client_socket.sendall(final_message.encode())

                self.client_socket.recvfrom()

                # print("Sending Message to Eval Client:", input_message)
            except Exception as _:
                traceback.print_exc()
                self.close_connection()


class Server(threading.Thread):
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
                message = self.connection.recv(64)
                # Append existing data into new data
                self.data = self.data + message

                if len(self.data) < 20:
                    continue
                packet = self.data[:20]
                self.data = self.data[20:]

                print("Message Received from Laptop:", packet)

                # Add to raw queue
                raw_queue.put(packet)
                fpga_queue.put(packet)

                game_state = laptop_queue.get()

                node_id = 0
                packet_type = PacketType.ACK
                header = (node_id << 4) | packet_type
                data = [header,
                        game_state['p1']['bullets'],
                        game_state['p1']['hp'],
                        0, 0, 0, 0, 0, 0, 0]

                self.connection.send(self.packer.pack(data))

                print('Sent back to laptop')
                if not message:
                    self.close_connection()
            except Exception as _:
                traceback.print_exc()
                self.close_connection()


class Training(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.shutdown = threading.Event()
        self.columns = ['flex1', 'flex2', 'gx', 'gy', 'gz', 'accX', 'accY', 'accZ']
        
        # defining headers for post processing
        # self.factors = ['mean', 'variance', 'median', 'root_mean_square', 'interquartile_range',            
        #     'percentile_75', 'kurtosis', 'min_max', 'signal_magnitude_area', 'zero_crossing_rate',            
        #     'spectral_centroid', 'spectral_entropy', 'spectral_energy', 'principle_frequency']
        self.factors = ['mean', 'std', 'variance', 'min', 'max', 'range', 'peak_to_peak_amplitude',
                         'mad', 'root_mean_square', 'interquartile_range', 'percentile_75',
                         'skewness', 'kurtosis', 'zero_crossing_rate', 'energy', 'entropy',
                         'kbest10_0', 'kbest10_1', 'kbest10_2', 'kbest10_3', 'kbest10_4', 
                         'kbest10_5', 'kbest10_6', 'kbest10_7', 'kbest10_8', 'kbest10_9',
                         'spectral_centroid', 'spectral_spread', 'wavelet_energy', 'wavelet_entropy']

        self.headers = [f'{raw_header}_{factor}' for raw_header in self.columns for factor in self.factors]
        self.headers.extend(['action', 'timestamp'])

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
        yaw = random.uniform(-180, 180)
        pitch = random.uniform(-180, 180)
        roll = random.uniform(-180, 180)
        accX = random.uniform(-1000, 1000)
        accY = random.uniform(-1000, 1000)
        accZ = random.uniform(-1000, 1000)
        flex1 = random.uniform(-180, 180)
        flex2 = random.uniform(-180, 180)
        return [flex1, flex2, yaw, pitch, roll, accX, accY, accZ]
    
    # simulate game movement with noise and action
    def generate_simulated_wave(self):

        # base noise 10s long -> 20Hz*10 = 200 samples
        t = np.linspace(0, 5, 200) # Define the time range
        x1 = 0.2 * np.sin(t) + 0.2 * np.random.randn(200) 
        x1[(x1 > -1) & (x1 < 1)] = 0.0 # TODO - sensor noise within margin of error auto remove
        
        # movement motion
        period = 2  # seconds
        amplitude = 5
        t = np.linspace(0, 2, int(2 / 0.05)) # Define the time range
        x2 = amplitude * np.sin(2 * np.pi * t / period)[:40] # Compute the sine wave for only one cycle

        x = x1 
        # Add to the 40th-80th elements
        x[20:60] += x2
        x[80:120] += x2

        return x


    def preprocess_data(self, data):
        data = data + 1e-12

        # Preprocess the data
        data_smoothed = sig.medfilt(data, kernel_size=3)
        # data_filtered = butter(3, 0.1, 'lowpass', fs=50)(data_smoothed)
        data_normalized = StandardScaler().fit_transform(data_smoothed.reshape(1, -1))

        # Extract features using Fourier transforms
        data_fft = np.abs(np.fft.fft(data_normalized, axis=1))
        data_fft = data_fft[:, :data_fft.shape[1]//2]

        # Extract features using wavelet transforms
        data_dwt = pywt.dwt(data_normalized, 'db1', axis=1)
        data_dwt = np.concatenate(data_dwt, axis=0)

        # Extract statistical features
        data_kurtosis = np.apply_along_axis(stats.kurtosis, axis=1, arr=data_normalized)
        data_skewness = np.apply_along_axis(stats.skew, axis=1, arr=data_normalized)
        data_entropy = np.apply_along_axis(stats.entropy, axis=1, arr=data_normalized)

        # Reshape statistical features to match shape of data_dwt
        data_kurtosis = np.tile(data_kurtosis, (2, 1)).T
        data_skewness = np.tile(data_skewness, (2, 1)).T
        data_entropy = np.tile(data_entropy, (2, 1)).T

        # Combine features into a feature matrix
        features = np.concatenate((data_fft, data_dwt.reshape(1, -1), data_kurtosis,
                                    data_skewness, data_entropy),
                                axis=1)

        # Replace NaN values with the mean of the column
        # features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))

        # Replace True/False values with 1/0
        features = features.astype(int)

        # Select top 10 features
        selector = SelectKBest(k=10)
        features_selected = selector.fit_transform(features, np.zeros(features.shape[0]))

        # Flatten to 1d array and return top 10 components
        top_10_components = features_selected.flatten()[:10]

        # Extract spectral centroid and spread
        data_normalized_mono = data_normalized[0]
        stft = librosa.stft(data_normalized_mono)

        # Extract spectral centroid and spread
        spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft), n_fft=64)
        spectral_spread = librosa.feature.spectral_bandwidth(S=np.abs(stft), n_fft=64)

        # Extract wavelet energy and entropy
        wavelet_coeffs = pywt.wavedec(data_normalized, 'db1', level=5)
        wavelet_energy = np.sum([np.sum(np.square(c)) for c in wavelet_coeffs])
        eps = 1e-10
        wavelet_entropy = np.sum([np.sum(np.square(c) * np.log(np.square(c) + eps)) for c in wavelet_coeffs])

        # Combine all features into a single array
        all_features = np.concatenate((top_10_components.reshape(-1, 1), spectral_centroid.reshape(-1, 1), spectral_spread.reshape(-1, 1),
                                        wavelet_energy.reshape(-1, 1), wavelet_entropy.reshape(-1, 1)), axis=0)

        # Return array of features
        all_features = all_features.reshape(1, -1)

        # standard data processing techniques
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
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        zero_crossing_rate = ((data[:-1] * data[1:]) < 0).sum()
        energy = np.sum(data**2)
        entropy = entropy(data, base=2)

        output_array = [mean, std, variance, min, max, range, peak_to_peak_amplitude,
                                mad, root_mean_square, interquartile_range, percentile_75,
                                skewness, kurtosis, zero_crossing_rate, energy, entropy]

        output_array = np.array(output_array)                        

        combined_array = np.concatenate((all_features.reshape(1, -1), output_array.reshape(1, -1)), axis=1)

        return combined_array
    
    def preprocess_dataset(self, df):
        processed_data = []

        # Loop through each column and compute features
        for column in df.columns:
            column_data = df[column]

            temp_processed = self.preprocess_data(column_data)

            # print(processed_column_data)
            # Append processed column data to main processed data array
            processed_data.append(temp_processed)

        processed_data_arr = np.concatenate(processed_data)
        
        # reshape into a temporary dataframe of 8x14
        temp_df = pd.DataFrame(processed_data_arr.reshape(8, -1), index=self.columns, columns=self.factors)

        # print the temporary dataframe
        print(f"processed_data: \n {temp_df} \n")

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


    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        unpacker = BLEPacket()
        all_data = []

        # live integration loop
        # while not self.shutdown.is_set():
        #
        #     df = pd.DataFrame(columns=['flex1', 'flex2', 'yaw', 'pitch', 'roll', 'accX', 'accY', 'accZ'])
        #     # Define the window size and threshold factor
        #     window_size = 11
        #     threshold_factor = 2
        #
        #     # Define N units for flagging movement, 20Hz -> 2s = 40 samples
        #     N = 40
        #
        #     # Initialize empty arrays for data storage
        #     t = []
        #     x = []
        #     filtered = []
        #     threshold = []
        #     movement_detected = []
        #     last_movement_time = -N  # set last movement time to negative N seconds ago
        #     wave = self.generate_simulated_wave()
        #     i = 0
        #     timenow = 0

            # print(f"entering while loop \n")

        # data collection loop
        i = 0
        while not self.shutdown.is_set():
            try:
                input("start?")

                # start_time = time.time()

                while i<41:
                    # getting data - simulation
                    data = self.generate_simulated_data()
                    print(f"data: {data} \n")

                    # # getting data - actl
                    # data = fpga_queue.get()
                    # unpacker.unpack(data)
                    # data = unpacker.get_flex_data() + unpacker.get_euler_data() + unpacker.get_acc_data()
                    # print(f"data: {data} \n")

                    if len(data) == 0:
                        print("Invalid data:", data)
                        continue
                    if len(data) == 8:
                        flex1, flex2, gx, gy, gz, accX, accY, accZ = data
                        all_data.append([flex1, flex2, gx, gy, gz, accX/100, accY/100, accZ/100])

                    self.sleep(0.05)
                    i += 1

                # creating df for preview 
                df = pd.DataFrame(all_data, columns=self.columns)
                # creating res to output differences 
                res = pd.DataFrame(columns=self.columns)

                for j in range(len(df)):
                    diff = df.iloc[j] - df.iloc[j-1]
                    res = res.append(diff, ignore_index=True)

                # Show user the data and prompt for confirmation
                print(res[['gx', 'gy', 'gz', 'accX', 'accY', 'accZ']].head(40))
                # print(f"Number of rows and columns: {df.shape[0]} by {df.shape[1]}")

                ui = input("data ok? y/n")
                if ui.lower() == "y":

                    time_now = time.strftime("%Y%m%d-%H%M%S")
                    
                    res_arr = res.values.reshape(1,-1)
                    res_arr = np.append(res_arr, time_now)

                    # Store data into a new CSV file
                    filename = "/home/xilinx/code/training/raw_data.csv"

                    with open(filename, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow(res_arr)

                    # Clear raw data list
                    all_data = []
                    res_arr = []
                    i = 0

                    # Preprocess data
                    processed_data = self.preprocess_dataset(res)

                    # Prompt user for label
                    label = input("Enter label (G = GRENADE, R = RELOAD, S = SHIELD, L = LOGOUT): ")

                    # Append label, timestamp to processed data
                    processed_data = np.append(processed_data, label)
                    processed_data = np.append(processed_data, time_now)

                    # Append processed data to CSV file
                    with open("/home/xilinx/code/training/processed_data.csv", "a") as f:
                        writer = csv.writer(f)
                        # writer.writerow(self.headers)
                        writer.writerow(processed_data)

                    print("Data processed and saved to CSV file.")
                else:
                    all_data = []
                    res_arr = 0
                    i = 0
                    print("not proceed, restart")
            except Exception as _:
                traceback.print_exc()
                self.close_connection()
                print("an error occurred")


if __name__ == '__main__':
    # Game Engine
    # print('---------------<Announcement>---------------')
    # print("Starting Game Engine Thread        ")
    # GE = GameEngine()
    # GE.start()

    # Software Visualizer Connection via Public Data Broker
    # print("Starting Subscriber Thread        ")
    # hive = Subscriber("CG4002")
    # hive.start()

    # # Client Connection to Evaluation Server
    # print("Starting Client Thread           ")
    # eval_client = EvalClient(1234, "localhost")
    # eval_client.start()

    # Server Connection to Laptop
    # print("Starting Server Thread           ")
    # laptop_server = Server(8080, "192.168.95.221")
    # laptop_server.start()

    # AI Model
    print("Starting AI Model Thread")
    ai_model = Training()
    ai_model.start()
    print('--------------------------------------------')
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
from fpga import training_data_collector
import csv
import pandas as pd
from intcomm import IntComm
import time
import numpy as np
from scipy import stats, signal
import csv
from ble_packet import BLEPacket

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
                print('Publishing to HiveMQ: ', input_message)
                if input_message == 'q':
                    break
                self.send_message(input_message)
            except Exception as _:
                traceback.print_exc()
                self.close_connection()


class Client(threading.Thread):
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

        print("Shutting Down Connection")

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

                self.client_socket.send(final_message.encode())

                print("Sending Message to Eval Client:", input_message)
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

        self.server_socket = server_socket
        self.connection = None
        self.secret_key = None
        self.secret_key_bytes = None

        # Flags
        self.shutdown = threading.Event()

    def setup(self):
        print('Awaiting Connection from Laptop')

        # Blocking Function
        self.connection, client_address = self.server_socket.accept()

        print('Successfully connected to', client_address[0])

        print('Setting up Secret Key')
        print('Default Secret Key: chrisisdabest123')
        secret_key = 'chrisisdabest123'

        if len(secret_key) == 16 or len(secret_key) == 24 or len(secret_key) == 32:
            # Store Secret Key and convert secret key into Byte Object
            self.secret_key = secret_key
            self.secret_key_bytes = bytes(str(secret_key), encoding="utf-8")
        else:
            self.close_connection()

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

                print("Message Received from Laptop:", message)

                # Add to raw queue
                raw_queue.put(message)
                fpga_queue.put(message)

                if not message:
                    self.close_connection()
            except Exception as _:
                traceback.print_exc()
                self.close_connection()

class Training(threading.Thread):
    def __init__(self):
        super().__init__()

    def init_csv(self):
        variables = ['Acc-X', 'Acc-Y', 'Acc-Z', 'Gyro-X', 'Gyro-Y', 'Gyro-Z', 'Flex1', 'Flex2']
        factors = ['mean', 'variance', 'median_absolute_deviation', 'root_mean_square', 'interquartile_range',
                   'percentile_75', 'kurtosis', 'min_max', 'signal_magnitude_area', 'zero_crossing_rate',
                   'spectral_centroid', 'spectral_entropy', 'spectral_energy', 'principle_frequency']

        headers = [f'{var}_{factor}' for var in variables for factor in factors]

        # Open a new CSV file and write headers
        with open('filename.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    def preprocess_data(df):
        def compute_mean(data):
            return np.mean(data)

        def compute_variance(data):
            return np.var(data)

        def compute_median_absolute_deviation(data):
            return stats.median_abs_deviation(data, axis=None)

        def compute_root_mean_square(data):
            return np.sqrt(np.mean(np.square(data)))

        def compute_interquartile_range(data):
            return stats.iqr(data)

        def compute_percentile_75(data):
            return np.percentile(data, 75)

        def compute_kurtosis(data):
            return stats.kurtosis(data)

        def compute_min_max(data):
            return np.max(data) - np.min(data)

        def compute_signal_magnitude_area(data):
            return np.sum(data) / len(data)

        def compute_zero_crossing_rate(data):
            return ((data[:-1] * data[1:]) < 0).sum()

        def compute_spectral_centroid(data):
            spectrum = np.abs(np.fft.rfft(data))
            normalized_spectrum = spectrum / np.sum(spectrum)
            normalized_frequencies = np.linspace(0, 1, len(spectrum))
            spectral_centroid = np.sum(normalized_frequencies * normalized_spectrum)
            return spectral_centroid

        def compute_spectral_entropy(data):
            freqs, power_density = signal.welch(data)
            return stats.entropy(power_density)

        def compute_spectral_energy(data):
            freqs, power_density = signal.welch(data)
            return np.sum(np.square(power_density))

        def compute_principle_frequency(data):
            freqs, power_density = signal.welch(data)
            return freqs[np.argmax(np.square(power_density))]

        processed_data = []

        # Loop through each column and compute features
        for column in df.columns:
            column_data = df[column]

            # Compute features for the column
            mean = compute_mean(column_data)
            variance = compute_variance(column_data)
            median_absolute_deviation = compute_median_absolute_deviation(column_data)
            root_mean_square = compute_root_mean_square(column_data)
            interquartile_range = compute_interquartile_range(column_data)
            percentile_75 = compute_percentile_75(column_data)
            kurtosis = compute_kurtosis(column_data)
            min_max = compute_min_max(column_data)
            signal_magnitude_area = compute_signal_magnitude_area(column_data)
            zero_crossing_rate = compute_zero_crossing_rate(column_data)
            spectral_centroid = compute_spectral_centroid(column_data)
            spectral_entropy = compute_spectral_entropy(column_data)
            spectral_energy = compute_spectral_energy(column_data)
            principle_frequency = compute_principle_frequency(column_data)

            # Store features in list
            processed_column_data = [mean, variance, median_absolute_deviation, root_mean_square,
                                     interquartile_range, percentile_75, kurtosis, min_max,
                                     signal_magnitude_area, zero_crossing_rate, spectral_centroid,
                                     spectral_entropy, spectral_energy, principle_frequency]
            print(processed_column_data)
            # Append processed column data to main processed data array
            processed_data.append(processed_column_data)

        processed_data_arr = np.concatenate(processed_data)

        return processed_data_arr

    def run(self):
        unpacker = BLEPacket()
        all_data = []
        print("Start")
        try:
            while True:
                # collecting data upon key press and 1s sleep timer
                input("Press any key to start data collection...")
                time.sleep(1)

                start_time = time.time()
                print("Recording for 1 second...")

                # assuming all actions within 1 second of key press
                while time.time() - start_time < 1:
                    data = fpga_queue.get()  # TODO comms - goal: get data from queue
                    data = unpacker.unpack(data)

                    print(data)
                    if len(data) == 0 or data[0] != "#":
                        print("Invalid data:", data)
                        continue

                    data = data[1:].split(",")
                    if len(data) == 8:
                        yaw, pitch, roll, accx, accy, accz, flex1, flex2 = data

                        all_data.append(
                            [yaw, pitch, roll, accx, accy, accz, flex1, flex2]
                        )

                # Convert data to DataFrame
                df = pd.DataFrame(all_data)
                df.columns = ["yaw", "pitch", "roll", "ax", "ay", "az", "flex1", "flex2"]

                # Show user the data and prompt for confirmation
                print(df.head())
                user_input = input("Does the data look ok? (y/n): ")
                if user_input.lower() == "y":
                    processed_data = self.preprocess_data(df)

                    # Append processed data to CSV file
                    with open("processed_data.csv", "a") as f:
                        writer = csv.writer(f)
                        writer.writerows(processed_data)

                    # Clear raw data list
                    raw_data = []
                    print("Data processed and saved to CSV file.")
                else:
                    print("Data not processed.")

        except KeyboardInterrupt:
            print("terminating program")
        except Exception:
            print("an error occured")




if __name__ == '__main__':
    # Game Engine
    print('---------------<Announcement>---------------')
    print("Starting Game Engine Thread        ")
    GE = GameEngine()
    GE.start()

    # Software Visualizer Connection via Public Data Broker
    print("Starting Subscriber Thread        ")
    hive = Subscriber("CG4002")
    hive.start()

    # Client Connection to Evaluation Server
    # print("Starting Client Thread           ")
    # eval_client = Client(1234, "localhost")
    # eval_client.start()

    # Server Connection to Laptop
    print("Starting Server Thread           ")
    laptop_server = Server(8080, "192.168.95.221")
    laptop_server.start()
    print('--------------------------------------------')

    # AI Model
    print("Starting AI Model Thread")
    ai_model = Training()
    ai_model.start()



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

                # Add-On for random actions
                # actions = ['shoot', 'reload', 'grenade', 'shield']
                # input_message = random.choice(actions)

                print('Action:', input_message)
                print(self.update(input_message))

                # Print out player status
                self.player.get_string()

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
        print('Publishing message: ' + message)
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

    def decrypt_message(self, message):
        # Decode from Base64 to Byte Object
        decode_message = base64.b64decode(message)
        # Initialization Vector
        iv = decode_message[:AES.block_size]

        # Create Cipher Object
        cipher = AES.new(self.secret_key_bytes, AES.MODE_CBC, iv)

        # Obtain Message using Cipher Decrypt
        decrypted_message_bytes = cipher.decrypt(decode_message[AES.block_size:])
        # Un-pad Message due to AES 16 bytes property
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
        self.server_socket.listen(1)
        self.setup()

        while not self.shutdown.is_set():
            try:
                # Receive up to 64 Bytes of data
                data = self.connection.recv(64)
                message = self.decrypt_message(data)

                print("Message Received from Laptop:", message)

                # Add to raw queue
                raw_queue.put(message)

                if not data:
                    self.close_connection()
            except Exception as _:
                traceback.print_exc()
                self.close_connection()


if __name__ == '__main__':
    # # Game Engine
    # print('---------------<Announcement>---------------')
    # print("Starting Game Engine Thread        ")
    # GE = GameEngine()
    # GE.start()
    #
    # # Software Visualizer Connection via Public Data Broker
    # print("Starting Subscriber Thread        ")
    # hive = Subscriber("CG4002")
    # hive.start()

    # Client Connection to Evaluation Server
    print("Starting Client Thread           ")
    eval_client = Client(1234, "localhost")
    eval_client.start()
    print('--------------------------------------------')

    # # Server Connection to Laptop
    # print("Starting Server Thread           ")
    # laptop_server = Server(8080, "192.168.95.221")
    # laptop_server = laptop_server.start()

import json

import paho.mqtt.client as mqtt
import socket
import base64
import sys
import threading
import traceback
from _socket import SHUT_RDWR
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad
from Crypto import Random


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
                input_message = input("Enter a message to publish (press 'q' to quit): ")
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
        # secret_key = self.client_socket.recv(1024).decode()

        self.secret_key = "chrisisdabest123"
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
                input_message = input("Enter a message to publish (press 'q' to quit): ")
                if input_message == 'q':
                    break
                with open('evaluation_server/example.json', 'r') as f:
                    json_string = json.dumps(f.read())
                print(json_string)
                encrypted_message = self.encrypt_message(json_string)
                # encrypted_message = self.encrypt_message(input_message)
                self.client_socket.send(encrypted_message.encode())
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
        print('Initializing Connection')

        # Blocking Function
        self.connection, client_address = self.server_socket.accept()

        print('Successfully connected to', client_address[0])

        print("Enter Secret Key: ")
        secret_key = sys.stdin.readline().strip()

        print('Connection from', client_address)
        if len(secret_key) == 16 or len(secret_key) == 24 or len(secret_key) == 32:
            # Send secret key to client
            self.connection.send(secret_key.encode())
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
        # Listen for ONE incoming connection
        self.server_socket.listen(1)
        self.setup()

        while not self.shutdown.is_set():
            try:
                data = self.connection.recv(1024)

                print("Decrypted Message: ", self.decrypt_message(data))

                if not data:
                    self.close_connection()
            except Exception as _:
                traceback.print_exc()
                self.close_connection()


class Connector(threading.Thread):
    def __init__(self):
        server_node = Server(8080, "127.0.0.1")
        client_node = Client(8080, "127.0.0.1")

        server_node.start()
        client_node.start()


if __name__ == '__main__':
    # hive = Subscriber("CG4002")
    # hive.start()

    # Client connection to Evaluation Server
    eval_client = Client(8080, "localhost")
    eval_client.start()


    # _parameters = 3
    # if len(sys.argv) != _parameters:
    #     print('---------------<Invalid number of arguments>---------------')
    #     print('python3 ' + os.path.basename(__file__) + ' [Port] [Hostname]')
    #     print('Port     : The port number for the TCP server')
    #     print('Hostname : Client HostName')
    #     print('Example  : python3 ' + os.path.basename(__file__) + ' 8080 0.0.0.0')
    #     print('-----------------------------------------------------------')
    #     sys.exit()
    # elif len(sys.argv) == _parameters:
    #     _port_num = int(sys.argv[1])
    #     _host_name = sys.argv[2]
    #
    #     client = Client(_port_num, _host_name)
    #     client.start()




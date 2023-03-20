import os
import socket
import base64
import sys
import threading
import traceback
from _socket import SHUT_RDWR

import time

class Client(threading.Thread):
    def __init__(self, port_num, host_name):
        super().__init__()

        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.client_socket = client_socket
        self.connection = client_socket.connect((host_name, port_num))

        # Flags
        self.shutdown = threading.Event()

    def close_connection(self):
        self.connection.shutdown(SHUT_RDWR)
        self.connection.close()
        self.shutdown.set()
        self.client_socket.close()

        print("Shutting Down Connection")

    def run(self):

        while not self.shutdown.is_set():
            try:
                message = input("Enter message to be sent: ")

                if message == 'q':
                    break

                self.client_socket.send(message.encode())
                # self.client_socket.send(message.encode())
                #
                # reply = self.client_socket.recv(64)
                #
                # print(reply)
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

                if not message:
                    self.close_connection()
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
                input_message = input("Message to be sent: ")
                if input_message == 'q':
                    break

                with open('example.json', 'r') as f:
                    json_data = f.read()

                encrypted_message = self.encrypt_message(input_message)

                # Format String for Eval Server Byte Sequence Process
                final_message = str(len(encrypted_message)) + "_" + encrypted_message

                self.client_socket.sendall(final_message.encode())

                self.client_socket.recvfrom()

                # print("Sending Message to Eval Client:", input_message)
            except Exception as _:
                traceback.print_exc()
                self.close_connection()


if __name__ == '__main__':
    _port_num = 8080
    _host_name = "localhost"

    # client = Client(8080, 'localhost')
    # client.start()
    client = Client(8080, 'localhost')
    client.start()
    # server = Server(8080, 'localhost')
    # server.start()

import base64
import os
import json
import socket
import sys
import threading
import traceback
from _socket import SHUT_RDWR
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from Crypto.Util.Padding import pad
from Crypto import Random


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

    def recv_game_state(self):
        game_state_received = None
        try:
            # recv length followed by '_' followed by cypher
            data = b''
            while not data.endswith(b'_'):
                _d = self.connection.recv(1)
                if not _d:
                    data = b''
                    break
                data += _d
            if len(data) == 0:
                print('no more data from the client')
                self.stop()

            data = data.decode("utf-8")
            length = int(data[:-1])

            data = b''
            while len(data) < length:
                _d = self.connection.recv(length - len(data))
                if not _d:
                    data = b''
                    break
                data += _d
            if len(data) == 0:
                print('no more data from the client')
                self.stop()
            msg = data.decode("utf8")  # Decode raw bytes to UTF-8
            game_state_received = self.decrypt_message(msg)
        except ConnectionResetError:
            print('Connection Reset')
            self.stop()
        return game_state_received

    def close_connection(self):
        self.connection.shutdown(SHUT_RDWR)
        self.connection.close()
        self.shutdown.set()

        print("Shutting Down Connection")

    def decrypt_message(self, cipher_text):
        decoded_message = base64.b64decode(cipher_text)                            # Decode message from base64 to bytes
        iv              = decoded_message[:AES.block_size]                         # Get IV value
        secret_key      = bytes(str(self.secret_key), encoding="utf8")             # Convert secret key to bytes

        cipher = AES.new(secret_key, AES.MODE_CBC, iv)                             # Create new AES cipher object

        decrypted_message = cipher.decrypt(decoded_message[AES.block_size:])       # Perform decryption
        decrypted_message = unpad(decrypted_message, AES.block_size)
        decrypted_message = decrypted_message.decode('utf8')                       # Decode bytes into utf-8

        print("Decrypted Message: ", decrypted_message)
        ret = json.loads(decrypted_message)
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
                data = self.recv_game_state()

                if not data:
                    self.close_connection()
            except Exception as _:
                traceback.print_exc()
                self.close_connection()


if __name__ == '__main__':
    _parameters = 3
    if len(sys.argv) != _parameters:
        print('---------------<Invalid number of arguments>---------------')
        print('python3 ' + os.path.basename(__file__) + ' [Port] [Hostname]')
        print('Port     : The port number for the TCP server')
        print('Hostname : Server HostName')
        print('Example  : python3 ' + os.path.basename(__file__) + ' 8080 0.0.0.0')
        print('-----------------------------------------------------------')
        sys.exit()
    elif len(sys.argv) == _parameters:
        _port_num = int(sys.argv[1])
        _host_name = sys.argv[2]

        server = Server(_port_num, _host_name)
        server.start()



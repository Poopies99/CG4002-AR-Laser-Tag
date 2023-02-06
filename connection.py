import base64
import os
import socket
import sys
import threading
import traceback
from _socket import SHUT_RDWR
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad


class Server(threading.Thread):
    def __init__(self, port_num, host_name):
        super().__init__()

        # Create a TCP/IP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Place Socket into TIME WAIT state
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Create Server Address
        server_address = (host_name, port_num)
        # Binds socket to specified host and port
        server_socket.bind(server_address)

        self.server_socket = server_socket
        self.connection = None
        self.secret_key = None

        # Flags
        self.shutdown = threading.Event()

    def setup(self):
        print('Initializing Connection')
        # Blocking Sockets
        self.connection, client_address = self.server_socket.accept()
        print('Successfully connected to', client_address[0])

        print("Enter Secret Key: ")
        secret_key = sys.stdin.readline().strip()

        print('Connection from', client_address)
        if len(secret_key) == 16 or len(secret_key) == 24 or len(secret_key) == 32:
            self.secret_key = secret_key
        else:
            self.stop()

    def stop(self):
        self.connection.shutdown(SHUT_RDWR)
        self.connection.close()
        self.shutdown.set()

        print("Shutting Down Connection")

    def decrypt_message(self, message):
        # Decode from Base64 to Byte Object
        decode_message = base64.b64decode(message)
        # Initialization Vector
        iv = decode_message[:AES.block_size]
        # Convert secret key into Byte Object
        secret_key = bytes(str(self.secret_key), encoding="utf-8")

        # Create Cipher Object
        cipher = AES.new(secret_key, AES.MODE_CBC, iv)

        # Obtain Message using Cipher Decrypt
        decrypted_message_bytes = cipher.decrypt(decode_message[AES.block_size:])
        # Unpad Message due to AES 16 bytes property
        decrypted_message_bytes = unpad(decrypted_message_bytes, AES.block_size)
        # Decode Bytes into utf-8
        decrypted_message = decrypted_message_bytes.decode("utf-8")

        return decrypted_message

    def run(self):
        # Listen for ONE incoming connection
        self.server_socket.listen(1)
        self.setup()

        while not self.shutdown.is_set():
            try:
                data = self.connection.recv(1024)
                print("Received: ", data)

                if not data:
                    self.stop()
            except Exception as _:
                traceback.print_exc()
                self.stop()


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



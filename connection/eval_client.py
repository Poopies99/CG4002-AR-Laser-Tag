import os
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


class Client(threading.Thread):
    def __init__(self, port_num):
        super().__init__()

        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.client_socket = client_socket
        self.connection = client_socket.connect(('', port_num))
        self.secret_key = None
        self.secret_key_bytes = None

        # Flags
        self.shutdown = threading.Event()

    def setup(self):
        print('Waiting for Secret Key')

        # Block Function
        secret_key = self.client_socket.recv(1024).decode()

        self.secret_key = secret_key
        self.secret_key_bytes = bytes(str(secret_key), encoding='utf-8')

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
                message = input("Enter message to be sent: ")
                if message == 'q':
                    break
                encrypted_message = self.encrypt_message(message)
                self.client_socket.send(encrypted_message.encode())
            except Exception as _:
                traceback.print_exc()
                self.close_connection()


if __name__ == '__main__':
    _parameters = 2
    if len(sys.argv) != _parameters:
        print('---------------<Invalid number of arguments>---------------')
        print('python3 ' + os.path.basename(__file__) + ' [Port] [Hostname]')
        print('Port     : The port number for the TCP server')
        print('Hostname : Client HostName')
        print('Example  : python3 ' + os.path.basename(__file__) + ' 8080 0.0.0.0')
        print('-----------------------------------------------------------')
        sys.exit()
    elif len(sys.argv) == _parameters:
        _port_num = int(sys.argv[1])

        client = Client(_port_num)
        client.start()

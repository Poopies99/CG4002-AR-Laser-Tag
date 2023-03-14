import threading
import socket
from _socket import SHUT_RDWR
import time


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
                message = self.connection.recv(64)
                print(message)

                self.connection.sendall(message)
            except:
                print('Server crash')


if __name__ == '__main__':
    server = Server(8080, 'localhost')
    server.start()


import threading
import socket
import traceback
from _socket import SHUT_RDWR
from external_comms.dependencies import constants
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

                print(f'Message Received: {message}'.ljust(80), end='\r')

                self.connection.sendall(message)
            except Exception as _:
                traceback.print_exc()
                print('Server crash')


class AnyOlHow(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        while True:
            print('Spamming Hello'.ljust(80), end='\r')

            time.sleep(2)


if __name__ == '__main__':
    server = Server(constants.XILINX_PORT_NUM, 'localhost')
    server.start()

    any = AnyOlHow()
    any.start()
    # server = Server(constants.XILINX_PORT_NUM, '192.168.95.221')
    # server.start()


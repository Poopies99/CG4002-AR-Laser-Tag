import threading
import traceback
import socket
import time
from _socket import SHUT_RDWR
import json

class ShootEngine(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.gun_shot = False
        self.vest_shot = False

    def handle_gun_shot(self):
        self.gun_shot = True

    def handle_vest_shot(self):
        self.vest_shot = True

    def run(self):
        while True:
            if self.gun_shot:
                self.gun_shot = False
                time.sleep(0.5)
                if self.vest_shot:
                    self.vest_shot = False
            elif self.vest_shot:
                self.vest_shot = False
                time.sleep(0.5)
                if self.gun_shot:
                    self.gun_shot = False


class Server(threading.Thread):
    shot_flag = False

    def __init__(self, port_num, host_name):
        super().__init__()

        # Create a TCP/IP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Place Socket into TIME WAIT state
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Binds socket to specified host and port
        server_socket.bind((host_name, port_num))

        self.server_socket = server_socket

        self.shoot_engine = ShootEngine()

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
        shot_thread = threading.Thread(target=self.shoot_engine.start)
        shot_thread.start()
        self.server_socket.listen(1)
        self.setup()

        while not self.shutdown.is_set():
            try:
                # Receive up to 64 Bytes of data
                data = int(self.connection.recv(64).decode())

                print(data)

                if data == 1:
                    self.shoot_engine.handle_gun_shot()
                elif data == 2:
                    self.shoot_engine.handle_vest_shot()
                else:
                    print("Invalid Beetle ID")

            except KeyboardInterrupt as _:
                traceback.print_exc()
                self.close_connection()
            except Exception as _:
                traceback.print_exc()
                continue


class ServerTwo(threading.Thread):
    def __init__(self, port_num, host_name):
        super().__init__()

        # Create a TCP/IP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Place Socket into TIME WAIT state
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Binds socket to specified host and port
        server_socket.bind((host_name, port_num))

        self.server_socket = server_socket

        # Data Buffer
        self.data = b''

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
        shot_thread = threading.Thread(target=self.shoot_engine.start)
        shot_thread.start()
        self.server_socket.listen(1)
        self.setup()

        while not self.shutdown.is_set():
            try:
                # Receive up to 64 Bytes of data
                data = self.connection.recv(64)

                with open('../external_comms/dependencies/example.json', 'r') as f:
                    data_json = json.loads(f.read())

                data = [0, data_json['p1']['bullets'], data_json['p1']['hp'], data_json['p2']['bullets'],
                        data_json['p2']['hp'], 0, 0, 0, 0, 0]

                self.connection.send(data)
            except KeyboardInterrupt as _:
                traceback.print_exc()
                self.close_connection()
            except Exception as _:
                traceback.print_exc()
                continue


if __name__ == '__main__':
    print('---------------<Setup Announcement>---------------')
    print("Starting Server Thread")
    laptop_server = Server(8080, 'localhost')
    print('--------------------------------------------------')

    laptop_server.start()
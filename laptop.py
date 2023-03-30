import socket
import threading
import traceback
from _socket import SHUT_RDWR

# from dependencies import constants
# from dependencies.ble_packet import BLEPacket

global_test = 2
print(global_test)

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
                reply = self.client_socket.recv(64)
                #
                print(reply)
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

        self.packer = BLEPacket
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

                if len(self.data) < constants.PACKET_SIZE:
                    continue
                packet = self.data[:constants.PACKET_SIZE]
                self.data = self.data[constants.PACKET_SIZE:]

                print("Message Received from Laptop:", packet)

                data = [0, 3, 20, 0, 4, 50, 0]
                data = self.packer.pack(data)

                self.connection.send(data)

                if not message:
                    self.close_connection()
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

    print("ehllo")
    # server = Server(8080, '192.168.95.221')
    # server.start()

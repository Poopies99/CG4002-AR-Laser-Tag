import threading
import socket
from _socket import SHUT_RDWR
import time
import traceback

BUFFER_SIZE = 512
DATA_TO_SEND = b'x' * 20 * 16 # 1 MB of data


class Client(threading.Thread):
    def __init__(self):
        super().__init__()

        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.client_socket = client_socket
        self.connection = client_socket.connect(("localhost", 8080))

        # Flags
        self.shutdown = threading.Event()

    def close_connection(self):
        self.connection.shutdown(SHUT_RDWR)
        self.connection.close()
        self.shutdown.set()
        self.client_socket.close()

        print("Shutting Down Connection")

    # run() function invoked by thread.start()
    def run(self):
        print('Client connected')
        res = []

        while not self.shutdown.is_set():
            try:
                input("start")

                for i in range(21):
                    start_time = time.time()

                    # Send the data to the server in chunks
                    total_sent = 0
                    while total_sent < len(DATA_TO_SEND):
                        data = DATA_TO_SEND[total_sent:]
                        sent = self.client_socket.send(data)
                        total_sent += sent

                    # Wait for the server to send the data back
                    received_data = b''
                    while len(received_data) < len(DATA_TO_SEND):
                        data = self.client_socket.recv(BUFFER_SIZE)
                        received_data += data

                    end_time = time.time()

                    # Calculate the time taken to send and receive the data
                    res.append(round((end_time - start_time), 4))
                print('Average Time Taken for Data transfer in {:.4f} seconds.'.format(sum(res) / len(res)))

            except Exception as _:
                traceback.print_exc()
                self.close_connection()


if __name__ == '__main__':
    client = Client()
    client.start()

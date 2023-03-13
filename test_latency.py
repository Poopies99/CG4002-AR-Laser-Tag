import threading
import socket
import time


class TCPClient(threading.Thread):
    def __init__(self, host, port, message):
        super().__init__()
        self.host = host
        self.port = port
        self.message = message
        self.shutdown = threading.Event()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            while not self.shutdown.is_set():
                s.sendall(self.message.encode())
                data = s.recv(1024)
                print('Received message:', data.decode())
                time.sleep(1)

    def stop(self):
        self.shutdown.set()


if __name__ == '__main__':
    client = TCPClient('localhost', 8080, 'Hello, world!')
    client.start()
    time.sleep(10)
    client.stop()

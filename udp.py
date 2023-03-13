import socket

class UDPServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self):
        self.sock.bind((self.host, self.port))
        print(f"Listening on {self.host}:{self.port}")
        while True:
            data, addr = self.sock.recvfrom(1024)
            print(f"Received data from {addr[0]}:{addr[1]}")
            if data:
                self.sock.sendto(data, addr)

if __name__ == "__main__":
    server = UDPServer("localhost", 8888)
    server.start()

import threading
import asyncio
import websockets
import time


class Server(threading.Thread):
    def __init__(self, port_num, host_name):
        super().__init__()

        self.port_num = port_num
        self.host_name = host_name

        # Flags
        self.shutdown = threading.Event()

    async def handle_connection(self, websocket, path):
        print("Connected to client at:", websocket.remote_address)

        # Wait for incoming data from the client
        async for message in websocket:
            print("Received message from client:", message)

            # Echo the message back to the client
            await websocket.send(message)
            print("Sent message to client:", message)

    async def start_server(self):
        async with websockets.serve(self.handle_connection, self.host_name, self.port_num):
            print("WebSocket server running on:", (self.host_name, self.port_num))
            await self.shutdown.wait()

    def run(self):
        asyncio.run(self.start_server())


class Client:
    async def send_data(self, websocket):
        message = "x" * 1024 * 1024 * 10  # 10 MB message
        start_time = time.monotonic()
        await websocket.send(message)
        await websocket.recv()
        end_time = time.monotonic()
        elapsed_time = end_time - start_time
        print(f"Sent and received 10 MB of data in {elapsed_time:.2f} seconds.")

    async def connect(self):
        async with websockets.connect("ws://localhost:8080") as websocket:
            await self.send_data(websocket)


if __name__ == "__main__":
    client = Client()
    asyncio.get_event_loop().run_until_complete(client.connect())


#
# if __name__ == '__main__':
#     # Create an instance of the Client class and connect to the remote server
#     client = Client("ws://localhost:8080")
#     client.start()

    # # Wait for some time
    # time.sleep(2)


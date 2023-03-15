import asyncio
import websockets

SERVER_ADDRESS = "ws://localhost:8080"
BUFFER_SIZE = 1024
DATA_TO_SEND = b'x' * 1024 * 1024 # 1 MB of data


class Client:
    def __init__(self, server_address):
        super().__init__()
        self.server_address = server_address

    async def send(self):
        async with websockets.connect(self.server_address) as websocket:
            print('Connected to server.')
            while True:
                try:
                    sent = await websocket.send("Hello".encode())
                except websockets.exceptions.ConnectionClosedError:
                    print('Server closed the connection prematurely.')
                    return

    async def run(self):
        await self.send()


if __name__ == "__main__":
    client = Client(SERVER_ADDRESS)
    asyncio.run(client.run())
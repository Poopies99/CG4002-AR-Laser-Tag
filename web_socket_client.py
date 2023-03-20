import asyncio
import websockets
import time

SERVER_ADDRESS = "ws://localhost:8080"
BUFFER_SIZE = 1024
DATA_TO_SEND = b'x' * 20 * 16 # 1 MB of data


async def start_client():
    async with websockets.connect(SERVER_ADDRESS) as websocket:
        res = []
        print('Client connected. Sending data...')

        for i in range(21):
            start_time = time.time()

            # Send the data to the server in chunks
            total_sent = 0
            while total_sent < len(DATA_TO_SEND):
                try:
                    # data = DATA_TO_SEND[total_sent:]
                    data = input("Input: ")
                    sent = await websocket.send(data)
                except websockets.exceptions.ConnectionClosedError:
                    print('Server closed the connection prematurely.')
                    return

                if not sent:
                    break
                total_sent += sent

            # Wait for the server to send the data back
            received_data = b''
            while len(received_data) < len(DATA_TO_SEND):
                data = await websocket.recv()
                if not data:
                    break
                received_data += data

            end_time = time.time()

            # Calculate the time taken to send and receive the data
            time_taken = end_time - start_time
            res.append(time_taken)

        print('Average Data sent and received in {:.4f} seconds.'.format(sum(res) / len(res)))

asyncio.run(start_client())

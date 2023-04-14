import asyncio
import websockets
from external_comms.dependencies import constants


async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(message)

async def start_server():
    async with websockets.serve(echo, constants.XILINX_SERVER, constants.XILINX_PORT_NUM):
        await asyncio.Future()

asyncio.run(start_server())

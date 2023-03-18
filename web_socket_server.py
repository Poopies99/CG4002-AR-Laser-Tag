import asyncio
import websockets
import constants


async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(message)

async def start_server():
    async with websockets.serve(echo, constants.xilinx_server, constants.xilinx_port_num):
        await asyncio.Future()

asyncio.run(start_server())

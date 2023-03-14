import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(message)

async def start_server():
    async with websockets.serve(echo, "192.168.95.221", 8080):
        await asyncio.Future()

asyncio.run(start_server())

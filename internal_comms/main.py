import sys

# File path of project directory
FILEPATH = '/home/kenneth/Desktop/CG4002/cg4002-internal-comms/'

# importing necessary module directories
sys.path.append(FILEPATH + 'bluno_beetle')
sys.path.append(FILEPATH + 'helper')

from bluno_beetle import BlunoBeetle
from bluno_beetle_game_state import BlunoBeetleGameState
from bluno_beetle_udp import BlunoBeetleUDP
from player import Player
from game_state import GameState
from _socket import SHUT_RDWR
from collections import deque

import constant
import socket
import threading
import traceback
import time

class Controller(threading.Thread):
    def __init__(self, params):
        super().__init__()
        
        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.client_socket = client_socket
        self.connection = client_socket.connect(("localhost" , 8080))
        self.secret_key = None
        self.secret_key_bytes = None

        # Flags
        self.shutdown = threading.Event()
        
        self.players = []
        for param in params:
            self.players.append(Player(param))
            Player.players_game_state.append(GameState())

        self.data = b''

    def close_connection(self):
        self.connection.shutdown(SHUT_RDWR)
        self.connection.close()
        self.shutdown.set()
        self.client_socket.close()

        print("Shutting Down Connection")
    
    def print_player_info(self):
        while True:
            for i in range(constant.STD_OP_LINES):
                print(constant.LINE_UP, end="")
            

            print("#" * constant.STD_OP_LENGTH)
            for player in self.players:
                player.print_statistics()
                print("#" * constant.STD_OP_LENGTH)

    def receive_game_state(self):
        while not self.shutdown.is_set():
            packet = self.client_socket.recv(16)
            
            #self.data = self.data + message

            #if len(self.data) < constant.PACKET_SIZE:
            #    continue

            #packet = self.data[:constant.PACKET_SIZE]
            #self.data = self.data[constant.PACKET_SIZE:]

            Player.update_game_state(packet)

    def run_threads(self):
        print_thread = threading.Thread(target=self.print_player_info, args=())
        receive_thread = threading.Thread(target=self.receive_game_state, args=()) 
        
        for player in self.players:
            player.start()
        
        print_thread.start()        
        receive_thread.start()
    
    # run() function invoked by thread.start()
    def run(self):
        for i in range(constant.STD_OP_LINES):
            print()

        self.run_threads()

        while not self.shutdown.is_set():
            try:
                if not BlunoBeetle.packet_queue:
                    continue
                
                data = BlunoBeetle.packet_queue.get()
                self.client_socket.send(data)
            except Exception as _:
                # traceback.print_exc()
                #self.close_connection()
                continue

if __name__ == '__main__':
    controller = Controller((
            (0,
            [1, constant.P1_IR_TRANSMITTER],    # P1 gun (IR transmitter)
            [2, constant.P1_IR_RECEIVER],       # P1 vest (IR receiver)
            [3, constant.P1_IMU_SENSOR]),       # P1 glove (IMU and flex sensors)
            (1,
            [4, constant.P2_IR_TRANSMITTER],    # P2 gun (IR transmitter)
            [5, constant.P2_IR_RECEIVER],       # P2 vest (IR receiver)
            [6, constant.P2_IMU_SENSOR])        # P2 glove (IMU and flex sensors)
            ))
    controller.start()

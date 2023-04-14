from bluno_beetle_udp import BlunoBeetleUDP
from bluno_beetle_game_state import BlunoBeetleGameState
from game_state import GameState
from ble_packet import BLEPacket

import time
import constant
import threading

class Player(threading.Thread):
    # arr of GameState objs to keep track of game state for all players
    # to be instantiated in main.py
    players_game_state = []

    def __init__(self, params):
        super().__init__()
        
        self.player_id = params[0]
        self.beetles = [    BlunoBeetleGameState([params[0]] + params[1]),    # gun (IR transmitter)
                            BlunoBeetleGameState([params[0]] + params[2]),    # vest (IR receiver)
                            BlunoBeetleUDP([params[0]] + params[3])]          # glove (imu_sensor)
        
        # statistics variables
        self.start_time = 0
        self.prev_time = 0
        self.prev_processed_bit_count = 0
        self.current_data_rate = 0
    
    @classmethod
    def update_game_state(cls, packet):
        unpacker = BLEPacket()
        unpacker.unpack(packet)
        p1_game_state = unpacker.get_euler_data()
        p2_game_state = unpacker.get_acc_data()
        cls.players_game_state[0].update_game_state(p1_game_state)
        cls.players_game_state[1].update_game_state(p2_game_state)
 
    # prints beetle data and statistics to std output
    def print_statistics(self):
        print("Player {} - Bullets = {}, Health = {}".ljust(constant.STD_OP_LENGTH).format(
            self.player_id + 1,
            Player.players_game_state[self.player_id].bullets,
            Player.players_game_state[self.player_id].health,
        ))
        #processed_bit_count = 0
        #fragmented_packet_count = 0
        for beetle in self.beetles:
            #processed_bit_count += beetle.get_processed_bit_count()
            #fragmented_packet_count += beetle.get_fragmented_packet_count()
            print("*" * constant.STD_OP_LENGTH)
            beetle.print_beetle_info()

        #print("*" * constant.STD_OP_LENGTH)
        #print("Statistics".ljust(constant.STD_OP_LENGTH))
        #current_time = time.perf_counter()
        #if current_time - self.prev_time >= 1:
        #    self.current_data_rate = ((processed_bit_count - self.prev_processed_bit_count) / 1000) / (current_time - self.prev_time)
        #    self.prev_time = current_time
        #    self.prev_processed_bit_count = processed_bit_count
        #print("Current data rate: {} kbps".ljust(constant.STD_OP_LENGTH).format(self.current_data_rate))
        #print("Average Data rate: {} kbps".ljust(constant.STD_OP_LENGTH).format(
        #    (processed_bit_count / 1000) / (current_time - self.start_time)
        #))
        #print("No. of fragmented packets: {}".ljust(constant.STD_OP_LENGTH).format(fragmented_packet_count))
        #print("************************************************************************************************************")

    def run(self): 
        self.start_time = time.perf_counter()

        for beetle in self.beetles:
            beetle.start()

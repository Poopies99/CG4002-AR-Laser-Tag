import sys
import os

# Module Dependencies Directory
sys.path.append('/home/xilinx/official/dependencies')

from dependencies import constants
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import tracemalloc
import threading
import traceback
import socket
import random
import time
import json
from queue import Queue
from _socket import SHUT_RDWR
from collections import deque
from player import PlayerAction
from GameState import GameState
from ble_packet import BLEPacket


import pynq
from scipy import stats
from scipy.fft import fft
from scipy.stats import moment
from pynq import Overlay


SINGLE_PLAYER_MODE = False
DEBUG_MODE = False

subscribe_queue = Queue()
feedback_queue = Queue()

raw_queue = deque()
eval_queue = deque()
shot_queue = deque()
fpga_queue = deque()
action_queue = deque()
laptop_queue = deque()
training_model_queue = deque()
ai_queue_1 = Queue()
ai_queue_2 = Queue()


class ActionEngine(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.p1_action_queue = deque()

        self.p1_gun_shot = False
        self.p1_vest_shot = False
        self.p1_grenade_hit = None

        if not SINGLE_PLAYER_MODE:
            self.p2_action_queue = deque()

            self.p2_gun_shot = False
            self.p2_vest_shot = False
            self.p2_grenade_hit = None

    def handle_grenade(self, player):
        print(f"Handling Grenade {player}")
        if player == 1:
            self.p1_action_queue.append('grenade')
        else:
            self.p2_action_queue.append('grenade')

    def handle_shield(self, player):
        print(f"Handling Shield {player}")
        if player == 1:
            self.p1_action_queue.append('shield')
        else:
            self.p2_action_queue.append('shield')

    def handle_reload(self, player):
        print(f"Handling Reload {player}")
        if player == 1:
            self.p1_action_queue.append('reload')
        else:
            self.p2_action_queue.append('reload')

    def handle_logout(self, player):
        print(f'Handling Logout {player}')
        if player == 1:
            self.p1_action_queue.append('logout')
        else:
            self.p2_action_queue.append('logout')

    def handle_gun_shot(self, player):
        print(f'Handling Gun Shot {player}')
        if player == 1:
            self.p1_gun_shot = True
            self.p1_action_queue.append('shoot')
        else:
            self.p2_gun_shot = True
            self.p2_action_queue.append('shoot')

    def handle_vest_shot(self, player):
        print(f'Handling Vest Shot {player}')
        if player == 1:
            self.p1_vest_shot = True
        else:
            self.p2_vest_shot = True
            
    def determine_grenade_hit(self, action_data_p1, action_data_p2):
        while True:
            while not feedback_queue.empty():
                data = feedback_queue.get()
                print(data)
                if data == "6 hit_grenade#":
                    self.p2_grenade_hit = True
                elif data == "3 hit_grenade#":
                    self.p1_grenade_hit = True
                elif data == "6 no#":
                    self.p2_grenade_hit = False
                elif data == "3 no#":
                    self.p1_grenade_hit = False                    
                
                if ((action_data_p1 == "grenade") and self.p2_grenade_hit is not None) and \
                    ((action_data_p2 == "grenade") and self.p1_grenade_hit is not None):
                        return
                
                if ((action_data_p1 == "grenade") and self.p2_grenade_hit is not None) and \
                    (action_data_p2 != "grenade"):
                    return
                    
                if ((action_data_p2 == "grenade") and self.p1_grenade_hit is not None) and \
                    (action_data_p1 != "grenade"):
                    return

    def run(self):
        action_data_p1, action_data_p2 = None, None
        action = [['None', True], ['None', True]]
        while True:
            if self.p1_action_queue or self.p2_action_queue:

                action_dic = {
                    "p1": {
                        "action": ""
                        },
                    "p2": {
                        "action": ""
                    } 
                }

                if action_data_p1 is None and self.p1_action_queue:
                    action_data_p1 = self.p1_action_queue.popleft()

                    if action_data_p1 == 'shoot':
                        action[0] = [action_data_p1, self.p2_vest_shot]
                    elif action_data_p1 == 'grenade':
                        action_dic["p1"]["action"] = "check_grenade"
                        action[0] = [action_data_p1, False]
                    else:
                        action[0] = [action_data_p1, True]

                if action_data_p2 is None and self.p2_action_queue:
                    action_data_p2 = self.p2_action_queue.popleft()

                    if action_data_p2 == 'shoot':
                        action[1] = [action_data_p2, self.p1_vest_shot]
                    elif action_data_p2 == 'grenade':
                        action_dic["p2"]["action"] = "check_grenade"
                        action[1] = [action_data_p2, False]
                    else:
                        action[1] = [action_data_p2, True]
                
                if action_data_p1 == "grenade" or action_data_p2 == "grenade":
                    subscribe_queue.put(json.dumps(action_dic))
                    # self.determine_grenade_hit(action_data_p1, action_data_p2)
                    print("done")
                    action[0][1] = self.p2_grenade_hit
                    action[1][1] = self.p1_grenade_hit
                    if action_data_p1 == "grenade":
                        # action_dic["p1"]["action"] = ""
                        action_data_p1 = False
                    if action_data_p2 == "grenade":
                        # action_dic["p2"]["action"] = ""
                        action_data_p2 = False
                        
                    
                if not (action_data_p1 is None or action_data_p2 is None):
                    action_queue.append(action)
                    action_data_p1, action_data_p2 = None, None
                    action = [['None', True], ['None', True]]

                    self.p1_grenade_hit = None
                    self.p1_gun_shot = False
                    self.p1_vest_shot = False
                    self.p1_action_queue.clear()

                    if not SINGLE_PLAYER_MODE:
                        self.p2_gun_shot = False
                        self.p2_vest_shot = False
                        self.p2_grenade_hit = None
                        self.p2_action_queue.clear()



                
class GameEngine(threading.Thread):
    def __init__(self, eval_client):
        super().__init__()

        # queue to receive status from sw
        self.eval_client = eval_client
        self.p1 = self.eval_client.gamestate.player_1
        self.p2 = self.eval_client.gamestate.player_2

        self.shutdown = threading.Event()

        self.p1_action = PlayerAction()

        if not SINGLE_PLAYER_MODE:
            self.p2_action = PlayerAction()


    def update_actions(self, player_action_value, player_action):
        if player_action_value == 'grenade':
            player_action.grenade()
        elif player_action_value == 'reload':
            player_action.reload()
        elif player_action_value == 'shield':
            player_action.shield()

    def reset_player(self, player):
        player.hp = 100
        player.bullets = 6
        player.grenades = 2
        player.shield_time = 0
        player.shield_health = 0
        player.num_shield = 3
        player.num_deaths += 1

    def run(self):
        action_counter = 0
        while not self.shutdown.is_set():
            try:
                if len(action_queue) != 0:
                    p1_action, p2_action = action_queue.popleft()
                    
                    action_dic = {
                        "p1": {
                            "action": ""
                        },
                        "p2": {
                            "action": ""
                        } 
                    }

                    if p1_action[0] != 'shoot' and not self.p1_action.check(p1_action[0]):
                        p1_action[0] = self.p1_action.secret_sauce()
                    if p2_action[0] != 'shoot' and not self.p1_action.check(p2_action[0]):
                        p2_action[0] = self.p2_action.secret_sauce()

                    viz_action_p1, viz_action_p2 = None, None

                    print(f"P1 action data Counter {action_counter}: {p1_action}")
                    print(f"P2 action data Counter {action_counter}: {p2_action}")
                
                    self.p1.update_shield()
                    self.p2.update_shield()

                    valid_action_p1 = self.p1.action_is_valid(p1_action[0])
                    valid_action_p2 = self.p2.action_is_valid(p2_action[0])

                    if valid_action_p1:
                        action_dic["p1"]["action"] = p1_action[0]
                    else:
                        action_dic["p1"]["action"] = p1_action[0] + "#"
                        
                    if valid_action_p2:
                        action_dic["p2"]["action"] = p2_action[0]
                    else:
                        action_dic["p2"]["action"] = p2_action[0] + "#"
                        
                    subscribe_queue.put(json.dumps(action_dic))
                    
                    if p1_action[0] == "logout" and p2_action[0] == "logout":
                        # send to visualizer
                        # send to eval server - eval_queue
                        data = self.eval_client.gamestate._get_data_plain_text()
                        subscribe_queue.put(data)
                        self.eval_client.submit_to_eval()
                        break
                    
                    if p1_action[0] == "shield":
                        if valid_action_p1 and not self.p1.check_shield():
                            self.p1.activate_shield()
                    
                    if p2_action[0] == "shield":
                        if valid_action_p2 and not self.p2.check_shield():
                            self.p2.activate_shield()
                    
                    if p1_action[0] == "grenade":
                        if valid_action_p1:
                            self.p1.throw_grenade()
                            if p1_action[1]:
                                viz_action_p2 = "hit_grenade"
                                self.p2.got_hit_grenade()
                                # if self.p2.check_shield():
                                #     viz_action_p2 = "hit_grenade_shield"
                                
                    if p2_action[0] == "grenade":
                        if valid_action_p2:
                            self.p2.throw_grenade()
                            if p2_action[1]:
                                viz_action_p1 = "hit_grenade"
                                self.p1.got_hit_grenade()
                                # if self.p1.check_shield():
                                #     viz_action_p1 = "hit_grenade_shield"
                    
                    if p1_action[0] == "shoot":
                        if valid_action_p1:
                            self.p1.shoot()
                            if p1_action[1]:
                                viz_action_p2 = "hit_bullet"
                                self.p2.got_shot()
                                # if self.p2.check_shield():
                                #     viz_action_p2 = "hit_shield"
                                
                    if p2_action[0] == "shoot":
                        if valid_action_p2:
                            self.p2.shoot()
                            if p2_action[1]:
                                viz_action_p1 = "hit_bullet"
                                self.p1.got_shot()
                                # if self.p1.check_shield():
                                #     viz_action_p1 = "hit_shield"
                                
                    if p1_action[0] == "reload":
                        if valid_action_p1:
                            self.p1.reload()
                                                                
                    if p2_action[0] == "reload":
                        if valid_action_p2:
                            self.p2.reload()
                            
                    if self.p1.hp <= 0:
                        self.reset_player(self.p1)
                    if self.p2.hp <= 0:
                        self.reset_player(self.p2)

                    self.p1.update_shield()
                    self.p2.update_shield()
                    
                    # gamestate to eval_server
                    self.eval_client.submit_to_eval()
                    # eval server to subscriber queue
                    correct_actions = self.eval_client.receive_correct_ans()
                    # If health drops to 0 then everything resets except for number of deaths

                    p1_action, p2_action = correct_actions['p1']['action'], correct_actions['p2']['action']
                    
                    valid_action_p1 = self.p1.action_is_valid(p1_action)
                    valid_action_p2 = self.p2.action_is_valid(p2_action)
                    
                    if p1_action == "shield":
                        if valid_action_p1 and not self.p1.check_shield():
                            self.p1.activate_shield()
                    
                    if p2_action == "shield":
                        if valid_action_p2 and not self.p2.check_shield():
                            self.p2.activate_shield()
                    
                    self.update_actions(p1_action, self.p1_action)
                    self.update_actions(p2_action, self.p2_action)

                    # subscriber queue to sw/feedback
                    self.p2.action = viz_action_p2                    
                    self.p1.action = viz_action_p1

                    laptop_queue.append(self.eval_client.gamestate._get_data_plain_text())
                    subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())

            except KeyboardInterrupt as _:
                traceback.print_exc()

class SubscriberSend(threading.Thread):
    def __init__(self, topic):
        super().__init__()

        # Create a MQTT client
        client = mqtt.Client()
        client.on_message = self.on_message

        self.client = client
        self.topic = topic

        # Flags
        self.shutdown = threading.Event()

    def setup(self):
        print('Setting up connection with HiveMQ')

        self.client.connect("broker.hivemq.com", 1883, 60)
        self.client.subscribe(self.topic)

        print('Successfully connected to HiveMQ and subscribed to topic: ' + self.topic)

    @staticmethod
    def on_message(msg):
        print('Received message: ' + msg.payload.decode())

    def close_connection(self):
        self.client.disconnect()
        self.shutdown.set()

        print("Shutting Down Connection to HiveMQ")

    def send_message(self, message):
        self.client.publish(self.topic, message)

    def run(self):
        self.setup()

        while not self.shutdown.is_set():
            try:
                if not subscribe_queue.empty():
                    input_message = subscribe_queue.get()

                    print('Publishing to HiveMQ: ', input_message)

                    self.send_message(input_message)

            except KeyboardInterrupt as _:
                traceback.print_exc()
                self.close_connection()
            except Exception as _:
                traceback.print_exc()
                continue


class SubscriberReceive(threading.Thread):
    def __init__(self, topic):
        super().__init__()

        # Create a MQTT client
        client = mqtt.Client()
        client.connect("broker.hivemq.com")
        client.subscribe(topic)

        client.on_message = self.on_message

        self.client = client
        self.topic = topic

        # Flags
        self.shutdown = threading.Event()

    @staticmethod
    def on_message(client, userdata, message):
        # print("Latency: %.4f seconds" % latency)
        # print('Received message: ' + message.payload.decode())
        feedback_queue.put(message.payload.decode())

    def close_connection(self):
        self.client.disconnect()
        self.shutdown.set()

        print("Shutting Down Connection to HiveMQ")

    def run(self):
        while not self.shutdown.is_set():
            try:
                self.client.loop_forever()

            except Exception as e:
                print(e)
                self.close_connection()


class EvalClient:
    def __init__(self, port_num, host_name):
        super().__init__()

        # Create a TCP/IP socket
        self.host_name = host_name
        self.port_num = port_num
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.secret_key = 'ilovecg4002profs'
        self.secret_key_bytes = bytes(str(self.secret_key), encoding='utf-8')

        # Create Player
        self.gamestate = GameState()

    def connect_to_eval(self):
        self.connection = self.client_socket.connect((self.host_name, self.port_num))
        print("[EvalClient] connected to eval server")

    def submit_to_eval(self):
        print(f'[EvalClient] Sent Game State: {self.gamestate._get_data_plain_text()}'.ljust(80), end='\r')
        self.gamestate.send_plaintext(self.client_socket)

    def receive_correct_ans(self):
        print(f'[EvalClient] Received Game State: {self.gamestate._get_data_plain_text()}'.ljust(80), end='\r')
        return self.gamestate.recv_and_update(self.client_socket)


    def close_connection(self):
        self.client_socket.close()
        print("Shutting Down EvalClient Connection")


class Server(threading.Thread):
    def __init__(self, port_num, host_name, action_engine_model):
        super().__init__()

        # Create a TCP/IP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Place Socket into TIME WAIT state
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Binds socket to specified host and port
        server_socket.bind((host_name, port_num))

        self.server_socket = server_socket

        # Shoot Engine Threads
        self.action_engine = action_engine_model

        # Flags
        self.shutdown = threading.Event()

    def setup(self):
        print('Awaiting Connection from Laptop')

        # Blocking Function
        self.connection, client_address = self.server_socket.accept()

        print('Successfully connected to', client_address[0])

    def close_connection(self):
        self.connection.shutdown(SHUT_RDWR)
        self.connection.close()
        self.shutdown.set()

        print("Shutting Down Server")

    def send_back_laptop(self, packer):
        game_state = json.loads(laptop_queue.popleft())
        data = [0, game_state['p1']['bullets'], game_state['p1']['hp'], 0, game_state['p2']['bullets'],
                game_state['p2']['hp'], 0, 0]
        data = packer.pack(data)

        self.connection.send(data)

        print("Sending back to laptop", data)

    def run(self):
        self.server_socket.listen(1)
        self.setup()

        while not self.shutdown.is_set():
            try:
                # Receive up to 64 Bytes of data
                packet = self.connection.recv(16)

                packer = BLEPacket()
                packer.unpack(packet)
                packet_id = packer.get_beetle_id()

                if packet_id == 1:
                    self.action_engine.handle_gun_shot(1)
                elif packet_id == 2:
                    self.action_engine.handle_vest_shot(1)
                elif packet_id == 3:
                    packet = packer.get_euler_data() + packer.get_acc_data()
                    ai_queue_1.put(packet)
                elif packet_id == 4:
                    self.action_engine.handle_gun_shot(2)
                elif packet_id == 5:
                    self.action_engine.handle_vest_shot(2)
                elif packet_id == 6:
                    packet = packer.get_euler_data() + packer.get_acc_data()
                    ai_queue_2.put(packet)
                else:
                    print("Invalid Beetle ID")

                if len(laptop_queue) != 0:
                    self.send_back_laptop(packer)
            except KeyboardInterrupt as _:
                traceback.print_exc()
                self.close_connection()
            except Exception as _:
                traceback.print_exc()
                continue


class AIModel(threading.Thread):
    def __init__(self, player, action_engine_model, queue_added, K):
        super().__init__()

        self.player = player
        self.action_engine = action_engine_model

        # Flags
        self.shutdown = threading.Event()

        # # Load in the features math model if we need vvv
        # features = np.load('dependencies/features_v1.5.6.npz', allow_pickle=True)
        # self.pca_eigvecs = features['pca_eigvecs']
        # self.weights = features['weights_list']
        # self.mean_vec = features['mean_vec']
        # self.scale = features['scale']
        # self.mean = features['mean']
        # # End of features math model ^^^

        self.K = K
        self.TOTAL_PACKET_COUNT = 30
        self.OVERLAY_INPUTS = 100
        self.OVERLAY_OUTPUTS = 4
        self.ai_queue = queue_added


        # # read in the test actions from the JSON file
        # with open('dependencies/test_actions.json', 'r') as f:
        #     test_actions = json.load(f)

        # # extract the test data for each action from the dictionary
        # self.test_g = np.array(test_actions['G'])
        # self.test_s = np.array(test_actions['S'])
        # self.test_r = np.array(test_actions['R'])
        # self.test_l = np.array(test_actions['L'])

        # # define the available actions
        # self.test_actions = ['G', 'S', 'R', 'L']

        # PYNQ overlay NEWEST - pca_mlp_1_5
        self.overlay = Overlay("dependencies/pca_mlp_1_5.bit")
        self.dma = self.overlay.axi_dma_0
        self.in_buffer = pynq.allocate(shape=(self.OVERLAY_INPUTS,), dtype=np.float32)
        self.out_buffer = pynq.allocate(shape=(self.OVERLAY_OUTPUTS,), dtype=np.float32)
        

        # PYNQ overlay NEW - pca_mlp_v3.5
#         self.overlay = Overlay("dependencies/pca_mlp_3_5.bit")
#         self.overlay.download()
#         self.dma = self.overlay.axi_dma_0
#         self.in_buffer = pynq.allocate(shape=(129,), dtype=np.float32)
#         self.out_buffer = pynq.allocate(shape=(3,), dtype=np.float32)

        # PYNQ overlay OLD backup - pca_mlp_1
        # self.overlay = Overlay("dependencies/pca_mlp_1.bit")
        # self.dma = self.overlay.axi_dma_0
        # self.in_buffer = pynq.allocate(shape=(125,), dtype=np.float32)
        # self.out_buffer = pynq.allocate(shape=(3,), dtype=np.float32)

    def sleep(self, seconds):
        start_time = time.time()
        while time.time() - start_time < seconds:
            pass
        
    def extract_features(self, raw_sensor_data):

        # Apply median filtering column-wise using the rolling function, window=5
        sensor_data = raw_sensor_data.rolling(5, min_periods=1, axis=0).mean()
        sensor_data = sensor_data.to_numpy()

        # Compute statistical features
        mean = np.mean(sensor_data, axis=0)
        std = np.std(sensor_data, axis=0)
        abs_diff = np.abs(np.diff(sensor_data, axis=0)).mean(axis=0)
        minimum = np.min(sensor_data, axis=0)
        maximum = np.max(sensor_data, axis=0)
        max_min_diff = maximum - minimum
        median = np.median(sensor_data, axis=0)
        mad = np.median(np.abs(sensor_data - np.median(sensor_data, axis=0)), axis=0)
        iqr = np.percentile(sensor_data, 75, axis=0) - np.percentile(sensor_data, 25, axis=0)
        negative_count = np.sum(sensor_data < 0, axis=0)
        positive_count = np.sum(sensor_data > 0, axis=0)
        values_above_mean = np.sum(sensor_data > mean, axis=0)
        
        peak_counts = np.array(np.apply_along_axis(lambda x: len(find_peaks(x)[0]), 0, sensor_data)).flatten()

        skewness = np.array(pd.DataFrame(sensor_data.reshape(-1,6)).skew().values).flatten()
        kurt = np.array(pd.DataFrame(sensor_data.reshape(-1,6)).kurtosis().values).flatten()
        energy = np.array(np.sum(sensor_data**2, axis=0)).flatten()

        # Compute the average resultant for gyro and acc columns
        gyro_cols = sensor_data[:, :3]
        acc_cols = sensor_data[:, 3:]
        gyro_avg_result = np.array(np.sqrt((gyro_cols**2).sum(axis=1)).mean()).flatten()
        acc_avg_result = np.array(np.sqrt((acc_cols**2).sum(axis=1)).mean()).flatten()

        # Compute the signal magnitude area for gyro and acc columns
        gyro_sma = np.array((np.abs(gyro_cols) / 100).sum(axis=0).sum()).flatten()
        acc_sma = np.array((np.abs(acc_cols) / 100).sum(axis=0).sum()).flatten()

        # Concatenate features and return as a list
        temp_features = np.concatenate([mean, std, abs_diff, minimum, maximum, max_min_diff, median, mad, iqr,
                                        negative_count, positive_count, values_above_mean, peak_counts, skewness, kurt, energy,
                                        gyro_avg_result, acc_avg_result, gyro_sma, acc_sma])
    
        return temp_features.tolist()

    # def rng_test_action(self):
    #     # choose a random action from the list
    #     chosen_action = random.choice(self.test_actions)

    #     # # print chosen action
    #     print(f'Chosen action: {chosen_action} \n')

    #     # use the chosen action to select the corresponding test data
    #     if chosen_action == 'G':
    #         test_data = self.test_g
    #     elif chosen_action == 'S':
    #         test_data = self.test_s
    #     elif chosen_action == 'L':
    #         test_data = self.test_l
    #     else:
    #         test_data = self.test_r

    #     return test_data

    # Define MLP
    def mlp_math(self, X):
        H1 = np.dot(X, self.weights[0]) + self.weights[1]
        H1_relu = np.maximum(0, H1)
        H2 = np.dot(H1_relu, self.weights[2]) + self.weights[3]
        H2_relu = np.maximum(0, H2)
        Y = np.dot(H2_relu, self.weights[4]) + self.weights[5]
        Y_softmax = np.exp(Y - np.max(Y)) / np.sum(np.exp(Y - np.max(Y)))
        return Y_softmax

    def get_action(self, softmax_array):
        max_index = np.argmax(softmax_array)
        action_dict = {0: 'G', 1: 'L', 2: 'R', 3: 'S'} 
        action = action_dict[max_index]
        return action

    # returns 1x4 softmax array
    def overlay_vivado(self, data):
        start_time = time.time()

        # reshape data to match in_buffer shape
        data = np.reshape(data, (self.OVERLAY_INPUTS,))

        self.in_buffer[:] = data

        self.dma.sendchannel.transfer(self.in_buffer)
        self.dma.recvchannel.transfer(self.out_buffer)

        # wait for transfer to finish
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # print output buffer
        print("mlp done with output: " + " ".join(str(x) for x in self.out_buffer))
        print(f"MLP time taken so far output: {time.time() - start_time}")

        return self.out_buffer

    def AIDriver(self, test_input):        
        sanity_data = test_input.reshape(1,-1)
        scaled_action_df = pd.DataFrame(sanity_data.reshape(-1,6))

        # 1. Feature extraction
        feature_vec = np.array(self.extract_features(scaled_action_df)).reshape(1,-1)

        # Vivado overlay does the below Steps 2-4
        pred_vivado = self.overlay_vivado(feature_vec)
        action_vivado = self.get_action(pred_vivado)

        print(pred_vivado)
        print(action_vivado)
        
        # # Math model does the below Steps 2-4 vvv
        # # 2. Scaler using features
        # scaled_action_math = (feature_vec - self.mean) / self.scale

        # # 3. PCA using scaler
        # pca_test_centered = scaled_action_math - self.mean_vec.reshape(1,-1)
        # pca_vec_math = np.dot(pca_test_centered, self.pca_eigvecs.T).astype(float)

        # # 4. MLP using PCA
        # pred_math = self.mlp_math(np.array(pca_vec_math).reshape(1,-1))
        # action_math = self.get_action(pred_math)

        # print(pred_math)
        # print(action_math)

        # # End of Math model ^^^

        return str(action_vivado)

    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        # Set the threshold value for movement detection based on user input
        # K = 5
        # K = float(input("threshold value? "))

        # Initialize arrays to hold the current and previous data packets
        current_packet = np.zeros((5, 6))
        previous_packet = np.zeros((5, 6))
        data_packet = np.zeros((self.TOTAL_PACKET_COUNT, 6))
        is_movement_counter = 0
        movement_watchdog = False
        loop_count = 0

        # live integration loop
        while True:
            if self.ai_queue:  # TODO re-enable for live integration
                # if 1 == 1: # TODO DIS-enable for live integration

                # runs loop to pack the data into groups of 5
                q_data = self.ai_queue.get()  # TODO re-enable for live integration
                self.ai_queue.task_done()  # TODO re-enable for live integration
                new_data = np.array(q_data)  # TODO re-enable for live integration
                new_data = new_data / 100.0  # TODO re-enable for live integration

                # new_data = np.random.randn(6) # TODO DIS-enable for live integration
                # print(" ".join([f"{x:.3f}" for x in new_data]))

                # Pack the data into groups of 6
                current_packet[loop_count] = new_data

                # Update loop_count
                loop_count = (loop_count + 1) % 5

                if loop_count % 5 == 0:
                    curr_mag = np.sum(np.square(np.mean(current_packet[:, -3:], axis=1)))
                    prev_mag = np.sum(np.square(np.mean(previous_packet[:, -3:], axis=1)))

                    # Check for movement detection
                    if not movement_watchdog and curr_mag - prev_mag > self.K:
                        print("Movement detected!")
                        # print currr and prev mag for sanity check
                        print(f"curr_mag: {curr_mag} \n")
                        print(f"prev_mag: {prev_mag} \n")
                        movement_watchdog = True
                        # append previous and current packet to data packet
                        data_packet = np.concatenate((previous_packet, current_packet), axis=0)

                    # movement_watchdog activated, count is_movement_counter from 0 up 6 and append current packet each time
                    if movement_watchdog:
                        if is_movement_counter < ((self.TOTAL_PACKET_COUNT - 10)/5):
                            data_packet = np.concatenate((data_packet, current_packet), axis=0)
                            is_movement_counter += 1

                        # If we've seen 6 packets since the last movement detection, preprocess and classify the data
                        else:
                            # print dimensions of data packet
                            # print(f"data_packet dimensions: {data_packet.shape} \n")

                            # rng_test_action = self.rng_test_action() # TODO DIS-enable for live integration
                            # action = self.AIDriver(rng_test_action) # TODO DIS-enable for live integration

                            action = self.AIDriver(data_packet)  # TODO re-enable for live integration
                            print(f"action from MLP in main: {action} \n")  # print output of MLP

                            if action == 'G':
                                self.action_engine.handle_grenade(self.player)
                            elif action == 'S':
                                self.action_engine.handle_shield(self.player)
                            elif action == 'R':
                                self.action_engine.handle_reload(self.player)
                            elif action == 'L':
                                self.action_engine.handle_logout(self.player)

                            # movement_watchdog deactivated, reset is_movement_counter
                            movement_watchdog = False
                            is_movement_counter = 0
                            # reset arrays to zeros
                            # current_packet = np.zeros((5, 6))
                            # previous_packet = np.zeros((5, 6))
                            data_packet = np.zeros((self.TOTAL_PACKET_COUNT, 6))

                    # Update the previous packet
                    previous_packet = current_packet.copy()


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Invalid number of arguments')
        print('Parameters: [num_of_players]')
        sys.exit()
    if int(sys.argv[1]) == 1:
        print("SINGLE PLAYER MODE")
        SINGLE_PLAYER_MODE = True
    else:
        print("TWO PLAYER MODE")

    if len(sys.argv) == 3 and sys.argv[2] == '-p':
        print('Debugging Mode Enabled')
        DEBUG_MODE = True

    print('---------------<Setup Announcement>---------------')
    # Action Engine
    print('Starting Action Engine Thread')
    action_engine = ActionEngine()
    action_engine.start()

    # Software Visualizer
    # print("Starting Subscriber Send Thread")
    # hive = SubscriberSend("CG4002")

    # Starting Visualizer Receive
    # print("Starting Subscribe Receive")
    # viz = SubscriberReceive("gamestate")

    # AI Model
    # ai_test = AIModel(1, [], [])
    # ai_test.start()

    ai_one = AIModel(1, action_engine, ai_queue_1, 5)
    ai_one.start()

    if not SINGLE_PLAYER_MODE:
        ai_two = AIModel(2, action_engine, ai_queue_2, 5)
        ai_two.start()

    # # Client Connection to Evaluation Server
    # print("Starting Client Thread")
    # # # eval_client = EvalClient(9999, "137.132.92.184")
    # eval_client = EvalClient(constants.EVAL_PORT_NUM, "localhost")
    # eval_client.connect_to_eval()

    # Game Engine
    # print("Starting Game Engine Thread")
    # game_engine = GameEngine(eval_client=eval_client)

    # # Server Connection to Laptop
    print("Starting Server Thread")
    laptop_server = Server(constants.XILINX_PORT_NUM, constants.XILINX_SERVER, action_engine)

    print('--------------------------------------------------')

    if not DEBUG_MODE:
        block_print()

    # hive.start()
    # viz.start()
    # game_engine.start()
    laptop_server.start()


    # tracemalloc.start()
    # start_time = time.time()
    # while True:
    #     if time.time() - start_time > 5:
    #         snapshot = tracemalloc.take_snapshot()
    #         top_stats = snapshot.statistics('lineno')
    #
    #         print("[ Top 10 ]")
    #         for stat in top_stats[:10]:
    #             print(stat)
    #
    #         start_time = time.time()



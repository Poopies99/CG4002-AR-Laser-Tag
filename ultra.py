import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import websockets
import threading
import traceback
import constants
import asyncio
import socket
import random
import joblib
import time
import json
import csv
import pynq
import librosa
import datetime
import queue
from pynq import Overlay
from GameState import GameState
from _socket import SHUT_RDWR
from scipy import stats
from queue import Queue
from collections import deque
from ble_packet import BLEPacket
from packet_type import PacketType
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

"""
Threads: 
1. Client Thread to connect to eval server
2. Server Thread for laptop to connect to
3. Subscriber Thread to publish data to SW Visualizer
4. Game Engine Thread to process incoming data from Laptop

Message Queues:
1. raw_queue between Server and Game Engine Thread
2. eval_queue from Game Engine to Client Thread
3. subscribe_queue from Game Engine to SW Visualizer
"""

raw_queue = deque()
action_queue = deque()
ai_queue = queue.Queue()
shot_queue = deque()
subscribe_queue = Queue()
fpga_queue = deque()
laptop_queue = deque()
training_model_queue = deque()
eval_queue = deque()
feedback_queue = Queue()

collection_flag = False


class ShootEngine(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.gun_shot = False
        self.vest_shot = False

    def handle_gun_shot(self):
        self.gun_shot = True

    def handle_vest_shot(self):
        self.vest_shot = True

    def run(self):
        while True:
            if self.gun_shot:
                self.gun_shot = False
                time.sleep(1)
                if self.vest_shot:
                    action_queue.append(['shoot', True])
                    self.vest_shot = False
                    print('Player has been shot')
                else:
                    action_queue.append(['shoot', False])
                    print('Player Missed')

            if self.vest_shot:
                self.vest_shot = False
                time.sleep(1)
                if self.gun_shot:
                    action_queue.append(['shoot', True])
                    self.gun_shot = False
                    print('Player has been shot')
                else:
                    action_queue.append(['shoot', False])
                    print('Player Missed')


class GameEngine(threading.Thread):
    def __init__(self, eval_client):
        super().__init__()

        # queue to receive status from sw
        self.eval_client = eval_client
        self.p1 = self.eval_client.gamestate.player_1
        self.p2 = self.eval_client.gamestate.player_2

        self.shutdown = threading.Event()

    def determine_grenade_hit(self):
        '''
        while True:
            print("Random")
            while not feedback_queue.empty():
                data = feedback_queue.get()
                if data == "6 hit_grenade#":
                    return True
                else:
                    return False
        '''
        return True
    # one approach is to put it in action queue and continue processing/ or do we want to wait for the grenade actions
    def random_ai_action(self, data):
        actions = ["shoot", "grenade", "shield", "reload", "invalid"]
        action_queue.put(([random.choice(actions)], ["False"]))

    def run(self):
        while not self.shutdown.is_set():
            try:
                if len(action_queue) != 0:
                    action_data, status = action_queue.popleft()
                    
                    print(f"Receive action data by Game Engine: {action_data}")

                
                    self.p1.update_shield()
                    self.p2.update_shield()

                    valid_action_p1 = self.p1.action_is_valid(action_data)

                    if valid_action_p1:
                        if action_data == "logout":
                            # send to visualizer
                            # send to eval server - eval_queue
                            data = self.eval_client.gamestate._get_data_plain_text()
                            subscribe_queue.put(data)
                            # self.eval_client.submit_to_eval()
                            break

                        if action_data == "grenade":
                        # receiving the status mqtt topic
                            self.p1.throw_grenade()
                            subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())

                        elif action_data == "shield":
                            self.p1.activate_shield()

                        elif action_data == "shoot":
                            self.p1.shoot()
                            if status:
                                self.p2.got_shot()

                        elif action_data == "reload":
                            self.p1.reload()

                        if action_data == "grenade":
                            if self.determine_grenade_hit():
                                self.p2.got_hit_grenade()

                    # If health drops to 0 then everything resets except for number of deaths
                    if self.p2.hp <= 0:
                        self.p2.hp = 100
                        self.p2.action = "none"
                        self.p2.bullets = 6
                        self.p2.grenades = 2
                        self.p2.shield_time = 0
                        self.p2.shield_health = 0
                        self.p2.num_shield = 3
                        self.p2.num_deaths += 1

                    # gamestate to eval_server
                    self.eval_client.submit_to_eval()
                    # eval server to subscriber queue
                    self.eval_client.receive_correct_ans()
                    # subscriber queue to sw/feedback queue

                    if valid_action_p1:
                        subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())
                    else:
                        self.p1.update_invalid_action()
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
        self.gamestate.recv_and_update(self.client_socket)

    def close_connection(self):
        self.client_socket.close()
        print("Shutting Down EvalClient Connection")


class Server(threading.Thread):
    def __init__(self, port_num, host_name):
        super().__init__()

        # Create a TCP/IP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Place Socket into TIME WAIT state
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Binds socket to specified host and port
        server_socket.bind((host_name, port_num))

        self.server_socket = server_socket

        self.packer = BLEPacket()

        self.shoot_engine = ShootEngine()

        # Data Buffer
        self.data = b''

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

    def run(self):
        p1_shot_thread = threading.Thread(target=self.shoot_engine.start)
        p1_shot_thread.start()
        self.server_socket.listen(1)
        self.setup()

        while not self.shutdown.is_set():
            try:
                # Receive up to 64 Bytes of data
                data = self.connection.recv(64)
                # Append existing data into new data
                self.data = self.data + data

                if len(self.data) < constants.packet_size:
                    continue

                packet = self.data[:constants.packet_size]
                self.data = self.data[constants.packet_size:]
                self.packer.unpack(packet)

                packet_id = self.packer.get_beetle_id()

                if packet_id == 1:
                    self.shoot_engine.handle_gun_shot()
                elif packet_id == 2:
                    self.shoot_engine.handle_vest_shot()
                elif packet_id == 3:
                    packet = self.packer.get_euler_data() + self.packer.get_acc_data()
                    ai_queue.put(packet)
                else:
                    print("Invalid Beetle ID")

                # # Remove when Training is complete
                # if global_flag:
                #     fpga_queue.put(packet)

                # Sends data back into the relay laptop
                if len(laptop_queue) != 0:
                    game_state = laptop_queue.popleft()
                    data = [0, game_state['p1']['bullets'], game_state['p1']['hp'], 0, game_state['p2']['bullets'],
                            game_state['p2']['hp'], 0]
                    data = self.packer.pack(data)

                    self.connection.send(data)

                    print("Sending back to laptop", data)
            except KeyboardInterrupt as _:
                traceback.print_exc()
                self.close_connection()
            except Exception as _:
                traceback.print_exc()
                continue


class AIModel(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.shutdown = threading.Event()

        # Load all_arrays.json
        with open('all_arrays.json', 'r') as f:
            all_arrays = json.load(f)

        # Retrieve values from all_arrays
        self.scaling_factors = np.array(all_arrays['scaling_factors'])
        self.mean = np.array(all_arrays['mean'])
        self.variance = np.array(all_arrays['variance'])
        self.pca_eigvecs = np.array(all_arrays['pca_eigvecs'])
        self.weights = [np.array(w) for w in all_arrays['weights']]

        # Reshape scaling_factors, mean and variance to (1, 3)
        self.scaling_factors = self.scaling_factors.reshape(40, 3)
        self.mean = self.mean.reshape(40, 3)
        self.variance = self.variance.reshape(40, 3)

        # read in the test actions from the JSON file
        with open('test_actions.json', 'r') as f:
            test_actions = json.load(f)

        # extract the test data for each action from the dictionary
        self.test_g = np.array(test_actions['G'])
        self.test_s = np.array(test_actions['S'])
        self.test_r = np.array(test_actions['R'])

        # define the available actions
        self.test_actions = ['G', 'S', 'R']
        
        # PYNQ overlay
        # self.overlay = Overlay("pca_mlp_1.bit")
        # self.dma = self.overlay.axi_dma_0

        # # Allocate input and output buffers once
        # self.in_buffer = pynq.allocate(shape=(35,), dtype=np.float32)
        # self.out_buffer = pynq.allocate(shape=(4,), dtype=np.float32)

    def sleep(self, seconds):
        start_time = time.time()
        while time.time() - start_time < seconds:
            pass
    
    def blur_3d_movement(self, acc_df):
        acc_df = pd.DataFrame(acc_df)
        fs = 20 # sampling frequency
        dt = 1/fs

        # Apply median filtering column-wise
        filtered_acc = acc_df.apply(lambda x: medfilt(x, kernel_size=7))
        filtered_acc = gaussian_filter1d(filtered_acc.values.astype(float), sigma=3, axis=0)
        filtered_acc_df = pd.DataFrame(filtered_acc, columns=acc_df.columns)
        
        ax = filtered_acc_df[0]
        ay = filtered_acc_df[1]
        az = filtered_acc_df[2]

        vx = np.cumsum(ax) * dt
        vy = np.cumsum(ay) * dt
        vz = np.cumsum(az) * dt

        x = np.cumsum(vx) * dt
        y = np.cumsum(vy) * dt
        z = np.cumsum(vz) * dt

        xyz = np.column_stack((x, y, z))

        return xyz
    
    # Define Scaler
    def scaler(self, X):
        return (X - self.mean) / np.sqrt(self.variance)

    # Define PCA
    def pca(self, X):
        return np.dot(X, self.pca_eigvecs.T)


    def rng_test_action(self):
        # choose a random action from the list
        chosen_action = random.choice(self.test_actions)

        # use the chosen action to select the corresponding test data
        if chosen_action == 'G':
            test_data = self.test_g
        elif chosen_action == 'S':
            test_data = self.test_s
        else:
            test_data = self.test_r

        return test_data


    # Define MLP
    def mlp(self, X):
        H1 = np.dot(X, self.weights[0]) + self.weights[1]
        H1_relu = np.maximum(0, H1)
        H2 = np.dot(H1_relu, self.weights[2]) + self.weights[3]
        H2_relu = np.maximum(0, H2)
        Y = np.dot(H2_relu, self.weights[4]) + self.weights[5]
        Y_softmax = np.exp(Y) / np.sum(np.exp(Y), axis=1, keepdims=True)
        return Y_softmax

    def get_action(self, softmax_array):
        max_index = np.argmax(softmax_array)
        action_dict = {0: 'G', 1: 'R', 2: 'S'}
        action = action_dict[max_index]
        return action


    # def MLP_Overlay(self, data):
    #     start_time = time.time()

    #     # reshape data to match in_buffer shape
    #     data = np.reshape(data, (35,))

    #     self.in_buffer[:] = data

    #     self.dma.sendchannel.transfer(self.in_buffer)
    #     self.dma.recvchannel.transfer(self.out_buffer)

    #     # wait for transfer to finish
    #     self.dma.sendchannel.wait()
    #     self.dma.recvchannel.wait()

    #     # print output buffer
    #     print("mlp done with output: " + " ".join(str(x) for x in self.out_buffer))

    #     print(f"MLP time taken so far output: {time.time() - start_time}")

    #     return self.out_buffer

    def AIDriver(self, test_input):
        test_input = test_input.reshape(40, 6)
        acc_df = test_input[:, -3:]
        
        # Transform data using Scaler and PCA
        blurred_data = self.blur_3d_movement(acc_df.reshape(40,3))
        data_scaled = self.scaler(blurred_data)
        data_pca = self.pca(data_scaled.reshape(1,120))

        # Make predictions using MLP
        predictions = self.mlp(data_pca)
        action = self.get_action(predictions)

        print(predictions)
        print(action)

        return action
        
    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        # Set the threshold value for movement detection based on user input
        K = 10
        # K = float(input("threshold value? "))

        # Initialize arrays to hold the current and previous data packets
        current_packet = np.zeros((5,6))
        previous_packet = np.zeros((5,6))
        data_packet = np.zeros((40,6))
        is_movement_counter = 0
        movement_watchdog = False

        # live integration loop
        while True:
            # runs loop 6 times and packs the data into groups of 6
            for i in range(5):
                new_data = np.random.randn(6) # TODO refactor for real data
                # print(" ".join([f"{x:.3f}" for x in new_data]))
            
                current_packet[i] = new_data
            
            curr_mag = np.sum(np.square(np.mean(current_packet[:, -3:], axis=1)))
            prev_mag = np.sum(np.square(np.mean(previous_packet[:, -3:], axis=1)))

            # Check for movement detection
            if not movement_watchdog and curr_mag - prev_mag > K:
                print("Movement detected!")
                # print currr and prev mag for sanity check
                print(f"curr_mag: {curr_mag} \n")
                print(f"prev_mag: {prev_mag} \n")
                movement_watchdog = True
                # append previous and current packet to data packet
                data_packet = np.concatenate((previous_packet, current_packet), axis=0)

            # movement_watchdog activated, count is_movement_counter from 0 up 6 and append current packet each time
            if movement_watchdog:
                if is_movement_counter < 6:
                    data_packet = np.concatenate((data_packet, current_packet), axis=0)
                    is_movement_counter += 1
                
                # If we've seen 6 packets since the last movement detection, preprocess and classify the data
                else:
                    # print dimensions of data packet
                    # print(f"data_packet dimensions: {data_packet.shape} \n")

                    # rng_test_action = self.rng_test_action() # TODO uncomment dummy data
                    # action = self.AIDriver(rng_test_action) # TODO uncomment dummy data

                    action = self.AIDriver(data_packet) # TODO uncomment for live integration
                    print(f"action from MLP in main: \n {action} \n")  # print output of MLP

                    # movement_watchdog deactivated, reset is_movement_counter
                    movement_watchdog = False
                    is_movement_counter = 0
                    # reset arrays to zeros
                    current_packet = np.zeros((5,6))
                    previous_packet = np.zeros((5,6))
                    data_packet = np.zeros((40,6))

            # Update the previous packet
            previous_packet = current_packet.copy()
            
            # except Exception as _:
            #     traceback.print_exc()
            #     self.close_connection()
            #     print("an error occurred")


class DetectionTime:
    def __init__(self):
        super().__init__()

        self.start = 0

    def start_timer(self):
        self.start = time.time()
        print('Starting Timer')

    def end_timer(self):
        end_time = time.time()
        print("Detection Time Taken: ", end_time - self.start)



class WebSocketServer:
    def __init__(self):
        super().__init__()

    async def receive(self, websocket):
        async for message in websocket:
            print(f'Message Received: {message}'.ljust(40), end='\r')
            await websocket.send(message.encode())

    async def start_server(self):
        async with websockets.serve(self.receive, constants.xilinx_server, constants.xilinx_port_num):
            await asyncio.Future()


if __name__ == '__main__':
    print('---------------<Setup Announcement>---------------')
    # AI Model
    print("Starting AI Model Thread")
    ai_model = AIModel()

    # Software Visualizer
    # print("Starting Subscriber Send Thread")
    # hive = SubscriberSend("CG4002")

    # Starting Visualizer Receive
    # print("Starting Subscribe Receive")
    # viz = SubscriberReceive("gamestate")

    # Client Connection to Evaluation Server
    # print("Starting Client Thread")
    # # eval_client = EvalClient(9999, "137.132.92.184")
    # eval_client = EvalClient(constants.eval_port_num, "localhost")
    # eval_client.connect_to_eval()

    # Game Engine
    # print("Starting Game Engine Thread")
    # game_engine = GameEngine(eval_client=eval_client)

    # Server Connection to Laptop
    print("Starting Server Thread")
    laptop_server = Server(constants.xilinx_port_num, constants.xilinx_server)

    # print("Starting Web Socket Server Thread")
    # server = WebSocketServer()
    # asyncio.run(server.start_server())

    print('--------------------------------------------------')

    # hive.start()
    # viz.start()
    # game_engine.start()
    ai_model.start()
    laptop_server.start()


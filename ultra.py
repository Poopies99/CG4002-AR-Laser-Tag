import sys

# Module Dependencies Directory
sys.path.append('/home/xilinx/official/dependencies')

from dependencies import constants
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import threading
import traceback
import socket
import random
import time
import json
import queue
from queue import Queue
from GameState import GameState
from _socket import SHUT_RDWR
from collections import deque
from ble_packet import BLEPacket
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter

# import pynq
# from scipy import stats
# from pynq import Overlay

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

SINGLE_PLAYER_MODE = False

raw_queue = deque()
action_queue = deque()
shot_queue = deque()
subscribe_queue = Queue()
fpga_queue = deque()
laptop_queue = deque()
training_model_queue = deque()
eval_queue = deque()
feedback_queue = Queue()


class ActionEngine(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.p1_gun_shot = False
        self.p1_vest_shot = False
        self.p1_grenade = False

        if not SINGLE_PLAYER_MODE:
            self.p2_gun_shot = False
            self.p2_vest_shot = False
            self.p2_grenade = False

    def handle_gun_shot(self, player):
        if player == 1:
            self.p1_gun_shot = True
        else:
            self.p2_gun_shot = True

    def handle_vest_shot(self, player):
        if player == 1:
            self.p1_vest_shot = True
        else:
            self.p2_vest_shot = True

    def handle_grenade_throw(self, player):
        if player == 1:
            self.p1_grenade = True
        else:
            self.p2_grenade = True

    def action_detected(self):
        return self.p1_gun_shot or self.p1_vest_shot or self.p1_grenade or self.p2_gun_shot or self.p2_vest_shot or self.p2_grenade
    
    def run(self):
        while True:
            if self.action_detected():
                time.sleep(1)

                action = [['None'], ['None']]
                # P1 action
                if self.p1_gun_shot:
                    action[0] = ['shoot', self.p2_vest_shot]
                    print('Player 1 has shot')
                    if self.p2_vest_shot:
                        print('Player 2 has been shot')
                    else:
                        print('Player 1 missed')
                if self.p2_gun_shot:
                    action[1] = ['shoot', self.p1_vest_shot]
                    print('Player 2 has shot')
                    if self.p1_vest_shot:
                        print('Player 1 has been shot')
                    else:
                        print('Player 2 missed')
                if self.p1_grenade:
                    # TODO - Check whether p2 is in frame
                    action[0] = ['grenade']
                if self.p2_grenade:
                    # TODO - Check whether p1 is in frame
                    action[1] = ['grenade']

                self.p1_gun_shot = False
                self.p1_vest_shot = False
                self.p1_grenade = False

                self.p2_gun_shot = False
                self.p2_vest_shot = False
                self.p2_grenade = False

                action_queue.append(action)


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
        action_queue.append(([random.choice(actions)], ["False"]))

    def run(self):
        while not self.shutdown.is_set():
            try:
                if len(action_queue) != 0:
                    p1_action, p2_action = action_queue.popleft() # [[p1_action, status], [p2_action, status]]
                    
                    print(f"P1 action data: {p1_action}")
                    print(f"P2 action data: {p2_action}")
                
                    self.p1.update_shield()
                    self.p2.update_shield()

                    valid_action_p1 = self.p1.action_is_valid(p1_action[0])
                    valid_action_p2 = self.p2.action_is_valid(p2_action[0])

                    if valid_action_p1:
                        if p1_action[0] == "logout":
                            # send to visualizer
                            # send to eval server - eval_queue
                            data = self.eval_client.gamestate._get_data_plain_text()
                            subscribe_queue.put(data)
                            # self.eval_client.submit_to_eval()
                            break
                        if p1_action[0] == "grenade":
                            # receiving the status mqtt topic
                            self.p1.throw_grenade()
                            subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())
                        elif p1_action[0] == "shield":
                            self.p1.activate_shield()
                        elif p1_action[0] == "shoot":
                            self.p1.shoot()
                            if p1_action[1]:
                                self.p2.got_shot()
                        elif p1_action[0] == "reload":
                            self.p1.reload()
                        if p1_action[0] == "grenade":
                            if self.determine_grenade_hit():
                                self.p2.got_hit_grenade()
                                
                    if valid_action_p2:
                        if p2_action[0] == "logout":
                            # send to visualizer
                            # send to eval server - eval_queue
                            data = self.eval_client.gamestate._get_data_plain_text()
                            subscribe_queue.put(data)
                            # self.eval_client.submit_to_eval()
                            break
                        if p2_action[0] == "grenade":
                            # receiving the status mqtt topic
                            self.p1.throw_grenade()
                            subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())
                        elif p2_action[0] == "shield":
                            self.p1.activate_shield()
                        elif p2_action[0] == "shoot":
                            self.p1.shoot()
                            if p2_action[1]:
                                self.p1.got_shot()
                        elif p2_action[0] == "reload":
                            self.p1.reload()
                        if p2_action[0] == "grenade":
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

                    if self.p1.hp <=0:
                        self.p1.hp = 100
                        self.p1.action = "none"
                        self.p1.bullets = 6
                        self.p1.grenades = 2
                        self.p1.shield_time = 0
                        self.p1.shield_health = 0
                        self.p1.num_shield = 3
                        self.p1.num_deaths += 1

                    # gamestate to eval_server
                    self.eval_client.submit_to_eval()
                    # eval server to subscriber queue
                    self.eval_client.receive_correct_ans()
                    # subscriber queue to sw/feedback queue

                    if valid_action_p1 or valid_action_p2:
                        subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())
                    else:
                        self.p1.update_invalid_action()
                        self.p2.update_invali_action()
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

        # Shoot Engine Threads
        self.action_engine = ActionEngine()

        # AI Model Threads
        self.p1_ai_engine = AIModel()
        if not SINGLE_PLAYER_MODE:
            self.p2_ai_engine = AIModel()

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
        action_thread = threading.Thread(target=self.action_engine.start)
        action_thread.start()

        p1_ai_thread = threading.Thread(target=self.p1_ai_engine.start)
        p1_ai_thread.start()

        if not SINGLE_PLAYER_MODE:
            p2_ai_thread = threading.Thread(target=self.p2_ai_engine.start)
            p2_ai_thread.start()

        self.server_socket.listen(1)
        self.setup()

        while not self.shutdown.is_set():
            try:
                # Receive up to 64 Bytes of data
                data = self.connection.recv(64)
                # Append existing data into new data
                self.data = self.data + data

                if len(self.data) < constants.PACKET_SIZE:
                    continue

                packet = self.data[:constants.PACKET_SIZE]
                self.data = self.data[constants.PACKET_SIZE:]
                self.packer.unpack(packet)

                packet_id = self.packer.get_beetle_id()

                print(packet)
                print("Packet ID: ", packet_id)

                if packet_id == 1:
                    self.action_engine.handle_gun_shot(1)
                elif packet_id == 2:
                    self.action_engine.handle_vest_shot(1)
                elif packet_id == 3:
                    packet = self.packer.get_euler_data() + self.packer.get_acc_data()
                    self.p1_ai_engine.add_packet(packet)
                elif packet_id == 4:
                    self.action_engine.handle_gun_shot(2)
                elif packet_id == 5:
                    self.action_engine.handle_vest_shot(2)
                elif packet_id == 6:
                    packet = self.packer.get_euler_data() + self.packer.get_acc_data()
                    self.p2_ai_engine.add_packet(packet)
                else:
                    print("Invalid Beetle ID")

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
        with open('dependencies/all_arrays.json', 'r') as f:
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
        with open('dependencies/test_actions.json', 'r') as f:
            test_actions = json.load(f)

        # extract the test data for each action from the dictionary
        self.test_g = np.array(test_actions['G'])
        self.test_s = np.array(test_actions['S'])
        self.test_r = np.array(test_actions['R'])
        self.test_l = np.array(test_actions['L'])

        # define the available actions
        self.test_actions = ['G', 'S', 'R', 'L']

        self.ai_queue = Queue()

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
        acc_df = acc_df.apply(pd.to_numeric)
        fs = 20  # sampling frequency
        dt = 1 / fs

        filtered_acc_df = acc_df.apply(lambda x: gaussian_filter(x, sigma=5))

        ax = filtered_acc_df[0]
        ay = filtered_acc_df[1]
        az = filtered_acc_df[2]

        vx = np.cumsum(ax) * dt
        vy = np.cumsum(ay) * dt
        vz = np.cumsum(az) * dt

        x = np.cumsum(vx) * dt
        y = np.cumsum(vy) * dt
        z = np.cumsum(vz) * dt

        x_arr = np.array(x)
        y_arr = np.array(y)
        z_arr = np.array(z)

        x_disp = x_arr[-1] - x_arr[0]
        y_disp = y_arr[-1] - y_arr[0]
        z_disp = z_arr[-1] - z_arr[0]

        xyz = np.column_stack((x, y, z))

        return xyz, [x_disp, y_disp, z_disp]

    def get_top_2_axes(self, row):
        row = np.array(row)
        abs_values = np.abs(row)
        top_2_idx = abs_values.argsort()[-2:][::-1]
        return (top_2_idx[0], top_2_idx[1])

    # Define Scaler
    def scaler(self, X):
        return (X - self.mean) / np.sqrt(self.variance)

    # Define PCA
    def pca(self, X):
        return np.dot(X, self.pca_eigvecs.T)

    def rng_test_action(self):
        # choose a random action from the list
        chosen_action = random.choice(self.test_actions)

        # # print chosen action
        # print(f'Chosen action: {chosen_action} \n')

        # use the chosen action to select the corresponding test data
        if chosen_action == 'G':
            test_data = self.test_g
        elif chosen_action == 'S':
            test_data = self.test_s
        elif chosen_action == 'L':
            test_data = self.test_l
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
        action_dict = {0: 'G', 1: 'L', 2: 'R', 3: 'S'}
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
        blurred_data, disp_change = self.blur_3d_movement(acc_df.reshape(40, 3))
        data_scaled = self.scaler(blurred_data)
        data_pca = self.pca(data_scaled.reshape(1, 120))

        top_2 = self.get_top_2_axes(disp_change)

        mlp_input = np.hstack((np.array(data_pca), np.array(disp_change).reshape(1, 3), np.array(top_2).reshape(1, 2)))

        # Make predictions using MLP
        predictions = self.mlp(mlp_input)
        action = self.get_action(predictions)

        print(predictions)
        print(action)

        return action

    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def add_packet(self, packet):
        self.ai_queue.put(packet)

    def run(self):
        # Set the threshold value for movement detection based on user input
        K = 10
        # K = float(input("threshold value? "))

        # Initialize arrays to hold the current and previous data packets
        current_packet = np.zeros((5, 6))
        previous_packet = np.zeros((5, 6))
        data_packet = np.zeros((40, 6))
        is_movement_counter = 0
        movement_watchdog = False
        loop_count = 0

        # live integration loop
        while True:
            if self.ai_queue:  # TODO re-enable for live integration
                # if 1 == 1: # TODO DIS-enable for live integration
                # runs loop 6 times and packs the data into groups of 6

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

                            # rng_test_action = self.rng_test_action() # TODO DIS-enable for live integration
                            # action = self.AIDriver(rng_test_action) # TODO DIS-enable for live integration

                            action = self.AIDriver(data_packet)  # TODO re-enable for live integration
                            print(f"action from MLP in main: {action} \n")  # print output of MLP

                            # movement_watchdog deactivated, reset is_movement_counter
                            movement_watchdog = False
                            is_movement_counter = 0
                            # reset arrays to zeros
                            current_packet = np.zeros((5, 6))
                            previous_packet = np.zeros((5, 6))
                            data_packet = np.zeros((40, 6))

                    # Update the previous packet
                    previous_packet = current_packet.copy()


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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Invalid number of arguments')
        print('Parameters: [num_of_players]')
        sys.exit()
    if int(sys.argv[1]) == 1:
        print("SINGLE PLAYER MODE")
        SINGLE_PLAYER_MODE = True
    else:
        print("TWO PLAYER MODE")

    print('---------------<Setup Announcement>---------------')
    # AI Model
    # print("Starting AI Model Thread")
    # ai_model = AIModel()

    # Software Visualizer
    # print("Starting Subscriber Send Thread")
    # hive = SubscriberSend("CG4002")

    # Starting Visualizer Receive
    # print("Starting Subscribe Receive")
    # viz = SubscriberReceive("gamestate")

    # Client Connection to Evaluation Server
    # print("Starting Client Thread")
    # # eval_client = EvalClient(9999, "137.132.92.184")
    # eval_client = EvalClient(constants.EVAL_PORT_NUM, "localhost")
    # eval_client.connect_to_eval()

    # Game Engine
    # print("Starting Game Engine Thread")
    # game_engine = GameEngine(eval_client=eval_client)

    # Server Connection to Laptop
    print("Starting Server Thread")
    laptop_server = Server(constants.XILINX_PORT_NUM, constants.XILINX_SERVER)

    # print("Starting Web Socket Server Thread")
    # server = WebSocketServer()
    # asyncio.run(server.start_server())

    print('--------------------------------------------------')

    # hive.start()
    # viz.start()
    # game_engine.start()
    # ai_model.start()
    laptop_server.start()


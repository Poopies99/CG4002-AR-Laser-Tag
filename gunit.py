import paho.mqtt.client as mqtt
import json
import socket
import base64
import threading
import traceback
import random
from _socket import SHUT_RDWR
from queue import Queue
import time
import numpy as np
from GameState import GameState

imu_queue = Queue()
action_queue = Queue()
feedback_queue = Queue()
subscribe_queue = Queue()  # GE and Subscriber
fpga_queue = Queue()


def sendActionDataProc():
    new_action_dataList = []
    # new_action_dataList.append(([' '], ["True"]))
    new_action_dataList.append((['shield'], ["True"]))
    new_action_dataList.append((['shoot'], ["True"]))
    new_action_dataList.append((["logout"], ["False"]))
    # new_action_dataList.append((['shoot'], ["False"]))
    # new_action_dataList.append((['grenade'], ["True"]))
    # new_action_dataList.append((['grenade'], ["True"]))
    # new_action_dataList.append((['shield'], ["True"]))
    # new_action_dataList.append((['grenade'], ["True"]))
    for i in range(len(new_action_dataList)):
        print(f'Inserting the following predicted action for Round {i}')
        print(new_action_dataList[i])
        action_queue.put(new_action_dataList[i])
        time.sleep(1)


# def populateIMUData():
#     new_imu_data = []
#     new_imu_data.append("1")
#     new_imu_data.append("5")

#     for i in range(len(new_imu_data)):
#         print(f'Inserting the following imu data for Reading {i}')
#         print(new_imu_data[i])
#         imu_queue.put(new_imu_data[i])
#         time.sleep(1)

class EvalClient:
    def __init__(self, host_name, port_num):
        super().__init__()

        # Create a TCP/IP socket
        self.host_name = host_name
        self.port_num = port_num
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.secret_key = 'chrisisdabest123'
        self.secret_key_bytes = bytes(str(self.secret_key), encoding='utf-8')

        # Create Player
        self.gamestate = GameState()

    def connect_to_eval(self):
        self.connection = self.client_socket.connect((self.host_name, self.port_num))
        print("[EvalClient] connected to eval server")

    def submit_to_eval(self):
        print(f"[EvalClient] Sending plain text gamestate data to the eval server")
        self.gamestate.send_plaintext(self.client_socket)

    def receive_correct_ans(self):
        print(f'[EvalClient] Received and update global gamestate')
        self.gamestate.recv_and_update(self.client_socket)

    def close_connection(self):
        self.client_socket.close()
        print("Shutting Down EvalClient Connection")


class GameEngine(threading.Thread):
    def __init__(self, eval_client):
        super().__init__()

        # queue to receive status from sw
        self.eval_client = eval_client
        self.p1 = self.eval_client.gamestate.player_1
        self.p2 = self.eval_client.gamestate.player_2

        self.shutdown = threading.Event()

    def determine_grenade_hit(self):
        while True:
            while not feedback_queue.empty():
                data = feedback_queue.get()
                if data == "6 hit_grenade#":
                    return True
                else:
                    return False

    # one approach is to put it in action queue and continue processing/ or do we want to wait for the grenade actions
    def random_ai_action(self, data):
        actions = ["shoot", "grenade", "shield", "reload", "invalid"]
        action_queue.put(([random.choice(actions)], ["False"]))

    def run(self):

        while not self.shutdown.is_set():
            try:

                if not action_queue.empty():
                    action_data, status = action_queue.get()

                    print(f"Receive action data by Game Engine: {action_data}")
                    # assuming action_data to be [[p1_action], [p2_status]]

                    if self.p1.shield_status:
                        self.p1.update_shield()

                    if self.p2.shield_status:
                        self.p2.update_shield()

                    if action_data == "logout":
                        self.p1.action = "logout"
                        # send to visualizer
                        # send to eval server - eval_queue
                        data = self.eval_client.gamestate._get_data_plain_text()
                        subscribe_queue.put(data)
                        # self.eval_client.submit_to_eval()
                        break

                    if action_data == "grenade":
                        # receiving the status mqtt topic

                        if self.p1.throw_grenade():
                            subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())
                            self.p1.action = "None"
                            # time.sleep(0.5)

                    elif action_data == "shield":
                        print("Entered shield action")
                        self.p1.activate_shield()

                    elif action_data == "grenade_hit_p2":
                        self.p2.got_hit_grenade()

                    elif action_data == "shoot":
                        print("Entered shoot action")
                        if self.p1.shoot() and status[0] == "True":
                            self.p2.got_shot()

                    elif action_data == "reload":
                        self.p1.reload()

                    if action_data == "grenade":
                        if self.p1.grenades >= 0:
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

                    subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())
                    time.sleep(1)

            except KeyboardInterrupt as _:
                traceback.print_exc()


class Subscriber(threading.Thread):
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
        # check for the message, and put only if it is "hit_grenade"
        feedback_queue.put(msg.payload.decode())

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

                    if input_message == 'q':
                        break
                    self.send_message(input_message)

                    # if not feedback_queue.empty():
                #     grenade_status = feedback_queue.get()
                #     print("grenade status" + grenade_status)

            except Exception as _:
                traceback.print_exc()
                self.close_connection()


class Viz_Subscriber(threading.Thread):
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
        print('Received message: ' + message.payload.decode())
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


if __name__ == '__main__':
    sendActionDataProc()
    hive = Subscriber("CG4002")
    hive.start()

    viz = Viz_Subscriber("gamestate")
    viz.start()

    eval_client = EvalClient(8080, "localhost")

    GE = GameEngine(eval_client=eval_client)
    GE.start()



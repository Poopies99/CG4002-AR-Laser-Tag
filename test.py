import sys
import os
import time
import json
import traceback
import threading
from collections import deque
import paho.mqtt.client as mqtt
from queue import Queue
SINGLE_PLAYER_MODE = False
action_queue = deque()

subscribe_queue = Queue()
feedback_queue = Queue()

class ActionEngine(threading.Thread):
    def __init__(self):
        super().__init__()

        self.p1_action_queue = deque()

        # Flags
        self.p1_gun_shot = False
        self.p1_vest_shot = False
        self.p1_grenade_hit = None

        if not SINGLE_PLAYER_MODE:
            self.p2_action_queue = deque()

            self.p2_gun_shot = False
            self.p2_vest_shot = False
            self.p2_grenade_hit = None

    def handle_grenade(self, player):
        print("Handling Grenade")
        if player == 1:
            self.p1_action_queue.append('grenade')
        else:
            self.p2_action_queue.append('grenade')

    def handle_shield(self, player):
        print("Handling Shield")
        if player == 1:
            self.p1_action_queue.append('shield')
        else:
            self.p2_action_queue.append('shield')

    def handle_reload(self, player):
        print("Handling Reload")
        if player == 1:
            self.p1_action_queue.append('reload')
        else:
            self.p2_action_queue.append('reload')

    def handle_logout(self, player):
        print('Handling Logout')
        if player == 1:
            self.p1_action_queue.append('logout')
        else:
            self.p2_action_queue.append('logout')

    def handle_gun_shot(self, player):
        print('Handling Gun Shot')
        if player == 1:
            self.p1_gun_shot = True
            self.p1_action_queue.append('shoot')
        else:
            self.p2_gun_shot = True
            self.p2_action_queue.append('shoot')

    def handle_vest_shot(self, player):
        print('Handling Vest Shot')
        if player == 1:
            self.p1_vest_shot = True
        else:
            self.p2_vest_shot = True
            
    def determine_grenade_hit(self, action_data_p1, action_data_p2):
        print("called determine grenade hit")
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

                    elif action_data_p1 == 'reload':
                        action[0] = [action_data_p1, True]

                    elif action_data_p1 == 'shield':
                        action[0] = [action_data_p1, True]

                    elif action_data_p1 == 'logout':
                        action[0] = [action_data_p1, True]

                if action_data_p2 is None and self.p2_action_queue:
                    action_data_p2 = self.p2_action_queue.popleft()

                    if action_data_p2 == 'shoot':
                        action[1] = [action_data_p2, self.p1_vest_shot]

                    elif action_data_p1 == 'grenade':
                        action_dic["p2"]["action"] = "check_grenade"
                        action[1] = [action_data_p2, False]

                    elif action_data_p2 == 'reload':
                        action[1] = [action_data_p2, True]

                    elif action_data_p2 == 'shield':
                        action[1] = [action_data_p2, True]

                    elif action_data_p2 == 'logout':
                        action[1] = [action_data_p2, True]

                if action_data_p1 is not None:
                    self.p1_action_queue.clear()

                if action_data_p2 is not None:
                    self.p2_action_queue.clear()
                
                if action_data_p1 == "grenade" or action_data_p2 == "grenade":
                    subscribe_queue.put(json.dumps(action_dic))
                    self.determine_grenade_hit(action_data_p1, action_data_p2)
                    print("done")
                    action[0][1] = self.p2_grenade_hit
                    action[1][1] = self.p1_grenade_hit
                    if action_data_p1 == "grenade":
                        action_dic["p1"]["action"] = ""
                        action_data_p1 = False
                    if action_data_p2 == "grenade":
                        action_dic["p2"]["action"] = ""
                        action_data_p2 = False
                        
                    print(action)
                    
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

if __name__ == '__main__':
    
    # Software Visualizer
    print("Starting Subscriber Send Thread")
    hive = SubscriberSend("CG4002")
    hive.start()
    
    # Starting Visualizer Receive
    print("Starting Subscribe Receive")
    viz = SubscriberReceive("gamestate")
    viz.start()
    
    action_engine = ActionEngine()
    action_engine.start()
    
    time.sleep(2)
    action_engine.handle_grenade(1)
    time.sleep(2)
    action_engine.handle_grenade(1)
    # action_engine.handle_gun_shot()
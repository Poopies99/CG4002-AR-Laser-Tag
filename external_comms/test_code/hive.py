# import paho.mqtt.client as mqtt
import threading
import traceback
from queue import Queue

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("CG4002")

def on_message(client, userdata, msg):
    print("Topic: " + msg.topic + " Message: " + str(msg.payload))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("broker.hivemq.com", 1883, 60)

client.publish('CG4002', "Hello")

client.loop_forever()

# subscribe_queue = Queue()
#
# class Subscriber(threading.Thread):
#     def __init__(self, topic):
#         super().__init__()
#
#         # Create a MQTT client
#         client = mqtt.Client()
#         client.on_message = self.on_message
#
#         self.client = client
#         self.topic = topic
#
#         # Flags
#         self.shutdown = threading.Event()
#
#     def setup(self):
#         print('Setting up connection with HiveMQ')
#
#         self.client.connect("broker.hivemq.com", 1883, 60)
#         self.client.subscribe(self.topic)
#
#         print('Successfully connected to HiveMQ and subscribed to topic: ' + self.topic)
#
#     def on_message(msg):
#         print('Raw Message: ', msg)
#         print('Received message: ' + msg.payload.decode())
#
#     def close_connection(self):
#         self.client.disconnect()
#         self.shutdown.set()
#
#         print("Shutting Down Connection to HiveMQ")
#
#     def send_message(self, message):
#         self.client.publish(self.topic, message)
#
#     def run(self):
#         self.setup()
#
#         while not self.shutdown.is_set():
#             try:
#                 self.client.loop()
#
#                 if not subscribe_queue.empty():
#                     input_message = subscribe_queue.get()
#                     # print('Publishing to HiveMQ: ', input_message)
#                     if input_message == 'q':
#                         break
#                     self.send_message(input_message)
#             except Exception as _:
#                 traceback.print_exc()
#                 self.close_connection()
#
#
# if __name__ == '__main__':
#     print("Starting Subscriber Thread")
#     hive = Subscriber("CG4002")
#     hive.start()
#
#     while True:
#         data = input('Message')
#         subscribe_queue.put(data)
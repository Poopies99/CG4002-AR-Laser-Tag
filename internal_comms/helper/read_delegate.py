from bluepy.btle import DefaultDelegate
from constant import PACKET_SIZE

class ReadDelegate(DefaultDelegate):
    def __init__(self):
        DefaultDelegate.__init__(self)
        self.data = b''

    def reset_buffer(self):
        self.data = b''

    def extract_buffer(self):
        packet = self.data[:PACKET_SIZE]
        self.data = self.data[PACKET_SIZE:]
        return packet

    def handleNotification(self, cHandle, data):
        self.data = self.data + data

    @property
    def buffer_len(self):
        return len(self.data)
 

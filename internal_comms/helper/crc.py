import struct
from constant import PACKET_SIZE

class CRC:
    def __init__(self):
        pass

    def calc(self, data):
    
        curr_crc = 0x0000
        sum1 = curr_crc
        sum2 = curr_crc >> 8

        for x in range(PACKET_SIZE):
            sum1 = (sum1 + data[x]) % 255
            sum2 = (sum2 + sum1) % 255

        return (sum2 << 8) | sum1

    #def check(self, packet):
    #    packet = list(struct.unpack(constant.PACKET_FORMAT_STRING, packet))
    #    crc = packet[9]
    #    packet[9] = 0
    #    packet = struct.pack(constant.PACKET_FORMAT_STRING, packet[0], 
    #            packet[1], packet[2], packet[3], 
    #            packet[4], packet[5], packet[6], 
    #            packet[7], packet[8], packet[9])
    #    return self.calc(packet) == crc



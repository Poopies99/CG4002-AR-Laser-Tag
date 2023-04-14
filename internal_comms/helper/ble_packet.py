from multipledispatch import dispatch
from constant import PACKET_FORMAT_STRING

import struct

class BLEPacket:
    def __init__(self):
        self.header = 0
        self.euler_x = 0
        self.euler_y = 0
        self.euler_z = 0
        self.acc_x = 0
        self.acc_y = 0
        self.acc_z = 0
        self.set_crc(0)
    
    #################### Packeting ####################

    @dispatch()
    def pack(self):
        return struct.pack(PACKET_FORMAT_STRING,
                self.header,
                self.euler_x,
                self.euler_y,
                self.euler_z,
                self.acc_x,
                self.acc_y,
                self.acc_z,
                self.crc)

    @dispatch(list)
    def pack(self, params):
        """
        self.update_attributes(params)
        return struct.pack(PACKET_FORMAT_STRING,
                self.header,
                self.euler_x,
                self.euler_y,
                self.euler_z,
                self.acc_x,
                self.acc_y,
                self.acc_z,
                self.crc)
        """
        return struct.pack(PACKET_FORMAT_STRING,
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
                params[6],
                params[7])

    # unpacks byte array and sets the attributes based on the packet data
    def unpack(self, packet):
        self.update_attributes(struct.unpack(PACKET_FORMAT_STRING, packet))
    
    #################### Getter functions ####################

    def get_header(self):
        return self.header

    def get_crc(self):
        return self.crc
     
    def get_beetle_id(self):
        return (self.header & 0xf0) >> 4
    
    def get_packet_type(self):
        return (self.header & 0b1100) >> 2

    # Used for new format
    def get_seq_no(self):
        return self.header & 0b1

    def get_euler_data(self):
        return [self.euler_x, self.euler_y, self.euler_z]

    def get_acc_data(self):
        return [self.acc_x, self.acc_y, self.acc_z]

    #################### Setter functions ####################

    def set_crc(self, new_crc):
        self.crc = new_crc
    
    def update_attributes(self, params):
        self.header = params[0]
        self.euler_x = params[1]
        self.euler_y = params[2]
        self.euler_z = params[3]
        self.acc_x = params[4]
        self.acc_y = params[5]
        self.acc_z = params[6]
        self.set_crc(params[7])


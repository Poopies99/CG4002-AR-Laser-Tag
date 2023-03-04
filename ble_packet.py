from multipledispatch import dispatch
import struct

class BLEPacket:
    format = "Bx8hH"

    def __init__(self):
        self.header = 0
        self.euler_x = 0
        self.euler_y = 0
        self.euler_z = 0
        self.acc_x = 0
        self.acc_y = 0
        self.acc_z = 0
        self.flex_1 = 0
        self.flex_2 = 0
        self.set_crc(0)

    def update_attributes(self, params):
        self.header = params[0]
        self.euler_x = params[1]
        self.euler_y = params[2]
        self.euler_z = params[3]
        self.acc_x = params[4]
        self.acc_y = params[5]
        self.acc_z = params[6]
        self.flex_1 = params[7]
        self.flex_2 = params[8]
        self.set_crc(params[9])
   
    @dispatch()
    def pack(self):
        return struct.pack(BLEPacket.format,
                self.header,
                self.euler_x,
                self.euler_y,
                self.euler_z,
                self.acc_x,
                self.acc_y,
                self.acc_z,
                self.flex_1,
                self.flex_2,
                self.crc)

    @dispatch(list)
    def pack(self, params):
        self.update_attributes(params)
        return struct.pack(BLEPacket.format,
                self.header,
                self.euler_x,
                self.euler_y,
                self.euler_z,
                self.acc_x,
                self.acc_y,
                self.acc_z,
                self.flex_1,
                self.flex_2,
                self.crc)

    # unpacks byte array and sets the attributes based on the packet data
    def unpack(self, packet):
        self.update_attributes(struct.unpack(BLEPacket.format, packet))

    def get_header(self):
        return self.header

    def get_crc(self):
        return self.crc
    
    def set_crc(self, new_crc):
        self.crc = new_crc

    def get_beetle_id(self):
        return (self.header & 0xf0) >> 4

    def get_packet_type(self):
        return self.header & 0xf

    def get_euler_data(self):
        return [self.euler_x, self.euler_y, self.euler_z]

    def get_acc_data(self):
        return [self.acc_x, self.acc_y, self.acc_z]

    def get_flex_data(self):
        return [self.flex_1, self.flex_2]

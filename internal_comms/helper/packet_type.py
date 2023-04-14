from enum import IntEnum

class PacketType(IntEnum):
    HELLO = 0
    ACK = 1
    NACK = 2
    DATA = 3

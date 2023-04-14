from bluno_beetle import BlunoBeetle
from packet_type import PacketType

class BlunoBeetleUDP(BlunoBeetle):
    def __init__(self, params):
        super().__init__(params)
    
    #################### Data processing ####################

    def process_data(self):
        packet = self.delegate.extract_buffer()
        self.ble_packet.unpack(packet)
        if self.crc_check() and self.packet_check(PacketType.DATA):
            self.queue_packet(packet)

            # for testing
            #self.print_test_data() 

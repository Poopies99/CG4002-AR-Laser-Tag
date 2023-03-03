# ONLY CAN RUN PYNQ LIB ON FPGA
import pynq
from pynq import Overlay
import numpy as np

overlay = Overlay("design_3.bit")

dma = overlay.axi_dma_0

# allocate in and out buffer
in_buffer = pynq.allocate(shape=(24,), dtype=np.double)

# out buffer of 1 integer
out_buffer = pynq.allocate(shape=(1,), dtype=np.int32)

# fill in buffer with data populate with array of [0.01, 0.02, .., 0.24]
input_data = np.arange(0.01, 0.25, 0.01, dtype=np.double)

for i, val in enumerate(input_data):
    in_buffer[i] = val

print(in_buffer)

# dma send and receive channel transfer
dma.sendchannel.transfer(in_buffer)
dma.recvchannel.transfer(out_buffer)

# wait for transfer to finish
dma.sendchannel.wait()
          
# print output buffer
for output in out_buffer:
    print(output)
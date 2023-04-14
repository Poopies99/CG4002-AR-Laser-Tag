import sys

# File path of project directory
FILEPATH = '/home/kenneth/Desktop/CG4002/cg4002-internal-comms/'

# importing necessary module directories
sys.path.append(FILEPATH + 'bluno_beetle')
sys.path.append(FILEPATH + 'helper')

from bluno_beetle import BlunoBeetle
from bluno_beetle_game_state import BlunoBeetleGameState

import constant

#ir_transmitter = BlunoBeetleGameState([0, 1, constant.P1_IR_TRANSMITTER])
#ir_transmitter = BlunoBeetle((4, constant.P2_IR_TRANSMITTER))
ir_transmitter = BlunoBeetleGameState([0, 1, constant.P1_IR_TRANSMITTER])
ir_transmitter.start()

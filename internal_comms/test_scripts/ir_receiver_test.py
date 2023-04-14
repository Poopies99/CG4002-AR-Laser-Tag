import sys

# File path of project directory
FILEPATH = '/home/kenneth/Desktop/CG4002/cg4002-internal-comms/'

# importing necessary module directories
sys.path.append(FILEPATH + 'bluno_beetle')
sys.path.append(FILEPATH + 'helper')

from bluno_beetle_game_state import BlunoBeetleGameState

import constant

ir_receiver = BlunoBeetleGameState([0, 2, constant.P1_IR_RECEIVER])
#ir_receiver = BlunoBeetle((5, constant.P2_IR_RECEIVER))
ir_receiver.start()

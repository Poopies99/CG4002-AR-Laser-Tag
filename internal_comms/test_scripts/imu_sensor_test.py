import sys

# File path of project directory
FILEPATH = '/home/kenneth/Desktop/CG4002/cg4002-internal-comms/'

# importing necessary module directories
sys.path.append(FILEPATH + 'bluno_beetle')
sys.path.append(FILEPATH + 'helper')

from bluno_beetle_udp import BlunoBeetleUDP

import constant

imu_sensor = BlunoBeetleUDP([0, 3, constant.P1_IMU_SENSOR])
#mu_sensor = BlunoBeetleUDP((3, constant.P2_IMU_SENSOR))
imu_sensor.start()

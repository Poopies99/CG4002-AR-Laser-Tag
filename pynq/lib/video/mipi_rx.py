#   Copyright (c) 2021, Xilinx, Inc.
#   All rights reserved.
# 
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pynq import DefaultIP

__author__ = "Mario Ruiz"
__copyright__ = "Copyright 2021, Xilinx"
__email__ = "pynq_support@xilinx.com"

_registers = {
	'core_configuration': {'address_offset': 0x00, 'access': 'read-write', 'size': 32, 'host_size': 4, 'description': 'The Core Configuration register', 'type': 'uint', 
		'fields': {
			'core_enabled': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 1, 'description': 'Core Enabled'}, 
			'soft_reset': {'access': 'read-write','bit_offset': 1,'bit_width': 1,'description': 'Soft Reset'}
		}
	},
	'protocol_configuration': {'address_offset': 0x04, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Protocol Configuration register', 'type': 'uint',
		'fields': {
			'active_lanes': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 2, 'description': 'Active Lanes'}, 
			'maximum_lanes': {'access': 'read-write','bit_offset': 3,'bit_width': 2,'description': 'Maximum Lane'}
		}
	},
	'core_status': {'address_offset': 0x10, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Core Status register', 'type': 'uint',
		'fields': {
			'soft_reset': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 1, 'description': 'Indicates that internal soft reset/core disable activities are in progress'},
			'stream_full': {'access': 'read-only', 'bit_offset': 1, 'bit_width': 1, 'description': 'Stream Line buffer Full: indicates the current status of line buffer full condition'},
			'shot_packet_fifo_not_empty': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': 'Short packet FIFO not empty: Indicates the current status of short packet FIFO not empty condition'},
			'shot_packet_fifo_full': {'access': 'read-only', 'bit_offset': 3, 'bit_width': 1, 'description': 'Short packet FIFO Full: FIFO full: Indicates the current status of short packet FIFO full condition'},
			'packet_count': {'access': 'read-only', 'bit_offset': 16, 'bit_width': 16, 'description': 'Counts number of long packets written to the line buffer'}
		}
	},
	'global_interrupt_enable': {'address_offset': 0x20, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Global Interrupt Enable register', 'type': 'uint',
		'fields': {
			'global_interrupt': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 1, 'description': 'Master enable for the device interrupt output to the system'}
		}
	},
	'interrupt_status': {'address_offset': 0x24, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Interrupt Status register', 'type': 'uint',
		'fields': {
			'Frame level error for VC0': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 1, 'description': 'Frame level error for VC0'},
			'Frame synchronization error for VC0': {'access': 'read-write', 'bit_offset': 1, 'bit_width': 1, 'description': 'Frame synchronization error for VC0'},
			'Frame level error for VC1': {'access': 'read-write', 'bit_offset': 2, 'bit_width': 1, 'description': 'Frame level error for VC1'},
			'Frame synchronization error for VC1': {'access': 'read-write', 'bit_offset': 3, 'bit_width': 1, 'description': 'Frame synchronization error for VC1'},
			'Frame level error for VC2': {'access': 'read-write', 'bit_offset': 4, 'bit_width': 1, 'description': 'Frame level error for VC2'},
			'Frame synchronization error for VC2': {'access': 'read-write', 'bit_offset': 5, 'bit_width': 1, 'description': 'Frame synchronization error for VC2'},
			'Frame level error for VC3': {'access': 'read-write', 'bit_offset': 6, 'bit_width': 1, 'description': 'Frame level error for VC3'},
			'Frame synchronization error for VC3': {'access': 'read-write', 'bit_offset': 7, 'bit_width': 1, 'description': 'Frame synchronization error for VC3'},
			'Unsupported Data Type': {'access': 'read-write', 'bit_offset': 8, 'bit_width': 1, 'description': 'Unsupported Data Type'},
			'CRC error': {'access': 'read-write', 'bit_offset': 9, 'bit_width': 1, 'description': 'CRC error'},
			'ECC 1-bit error': {'access': 'read-write', 'bit_offset': 10, 'bit_width': 1, 'description': 'ECC 1-bit error'},
			'ECC 2-bit error': {'access': 'read-write', 'bit_offset': 11, 'bit_width': 1, 'description': 'ECC 2-bit error'},
			'SoT sync error': {'access': 'read-write', 'bit_offset': 12, 'bit_width': 1, 'description': 'SoT sync error'},
			'SoT error': {'access': 'read-write', 'bit_offset': 13, 'bit_width': 1, 'description': 'SoT error'},
			'stop_state': {'access': 'read-write', 'bit_offset': 17, 'bit_width': 1, 'description': 'Active-High signal indicates that the lane module is currently in Stop state'},
			'stream line buffer full ': {'access': 'read-write', 'bit_offset': 18, 'bit_width': 1, 'description': 'Asserts when the line buffer is full'},
			'Short packet FIFO not empty': {'access': 'read-write', 'bit_offset': 19, 'bit_width': 1, 'description': 'Active-High signal asserted when short packet FIFO not empty condition detected'},
			'Short packet FIFO full': {'access': 'read-write', 'bit_offset': 20, 'bit_width': 1, 'description': 'Active-High signal asserted when the short packet FIFO full condition detected'},
			'Incorrect lane configuration': {'access': 'read-write', 'bit_offset': 21, 'bit_width': 1, 'description': 'Asserted when Active lanes is greater than Maximum lanes in the protocol configuration register'},
			'Word Count corruption': {'access': 'read-write', 'bit_offset': 22, 'bit_width': 1, 'description': 'Asserted when WC field of packet header corrupted and core receives less bytes than indicated in WC field'},
			'UV420 WC Error': {'access': 'read-write', 'bit_offset': 28, 'bit_width': 1, 'description': 'Asserted when the user-configured YUV420 word count value is less than the actual Y-line word count of the incoming data, which results in an internal buffer full condition.'},
			'RX_Skewcalhs': {'access': 'read-write', 'bit_offset': 29, 'bit_width': 1, 'description': 'Asserted when rxskewcalhs is detected.'},
			'VCX Frame Error': {'access': 'read-only', 'bit_offset': 30, 'bit_width': 1, 'description': 'Asserted when the VCX Frame error is detected'},
			'Frame Received': {'access': 'read-write', 'bit_offset': 31, 'bit_width': 1, 'description': 'Asserted when the Frame End (FE) short packet is received for the current frame'}
		}
	},
	'interrupt_enable': {'address_offset': 0x28, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Interrupt Enable register', 'type': 'uint',
		'fields': {
			'Frame level error for VC0': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Frame synchronization error for VC0': {'access': 'read-write', 'bit_offset': 1, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Frame level error for VC1': {'access': 'read-write', 'bit_offset': 2, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Frame synchronization error for VC1': {'access': 'read-write', 'bit_offset': 3, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Frame level error for VC2': {'access': 'read-write', 'bit_offset': 4, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Frame synchronization error for VC2': {'access': 'read-write', 'bit_offset': 5, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Frame level error for VC3': {'access': 'read-write', 'bit_offset': 6, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Frame synchronization error for VC3': {'access': 'read-write', 'bit_offset': 7, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Unsupported Data Type': {'access': 'read-write', 'bit_offset': 8, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'CRC error': {'access': 'read-write', 'bit_offset': 9, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'ECC 1-bit error': {'access': 'read-write', 'bit_offset': 10, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'ECC 2-bit error': {'access': 'read-write', 'bit_offset': 11, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'SoT sync error': {'access': 'read-write', 'bit_offset': 12, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'SoT error': {'access': 'read-write', 'bit_offset': 13, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'stop_state': {'access': 'read-write', 'bit_offset': 17, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'stream line buffer full ': {'access': 'read-write', 'bit_offset': 18, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Short packet FIFO not empty': {'access': 'read-write', 'bit_offset': 19, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Short packet FIFO full': {'access': 'read-write', 'bit_offset': 20, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Incorrect lane configuration': {'access': 'read-write', 'bit_offset': 21, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Word Count corruption': {'access': 'read-write', 'bit_offset': 22, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'UV420 WC Error': {'access': 'read-write', 'bit_offset': 28, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'RX_Skewcalhs': {'access': 'read-write', 'bit_offset': 29, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'VCX Frame Error': {'access': 'read-write', 'bit_offset': 30, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'},
			'Frame Received': {'access': 'read-write', 'bit_offset': 31, 'bit_width': 1, 'description': 'Set bits in this register to 1 to generate the required interrupts'}
		}
	},
	'generic_short_packet': {'address_offset': 0x30, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Generic Short Packet register', 'type': 'uint',
		'fields': {
			'data_type': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 6, 'description': 'Generic short packet code'},
			'virtual_channel': {'access': 'read-only', 'bit_offset': 6, 'bit_width': 2, 'description': 'Virtual channel number'},
			'data': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': '16-bit short packet data'}
		}
	},
	'vcx_frame_error': {'address_offset': 0x34, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The VCX Frame Error register', 'type': 'uint',
		'fields': {
			'Fame level error for VC4': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC4': {'access': 'read-write', 'bit_offset': 1, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC5': {'access': 'read-write', 'bit_offset': 2, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC5': {'access': 'read-write', 'bit_offset': 3, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC6': {'access': 'read-write', 'bit_offset': 4, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC6': {'access': 'read-write', 'bit_offset': 5, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC7': {'access': 'read-write', 'bit_offset': 6, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC7': {'access': 'read-write', 'bit_offset': 7, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC8': {'access': 'read-write', 'bit_offset': 8, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC8': {'access': 'read-write', 'bit_offset': 9, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC9': {'access': 'read-write', 'bit_offset': 10, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC9': {'access': 'read-write', 'bit_offset': 11, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC10': {'access': 'read-write', 'bit_offset': 12, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC10': {'access': 'read-write', 'bit_offset': 13, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC11': {'access': 'read-write', 'bit_offset': 14, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC11': {'access': 'read-write', 'bit_offset': 15, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC12': {'access': 'read-write', 'bit_offset': 16, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC12': {'access': 'read-write', 'bit_offset': 17, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC13': {'access': 'read-write', 'bit_offset': 18, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC13': {'access': 'read-write', 'bit_offset': 19, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC14': {'access': 'read-write', 'bit_offset': 20, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC14': {'access': 'read-write', 'bit_offset': 21, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'},
			'Fame level error for VC15': {'access': 'read-write', 'bit_offset': 22, 'bit_width': 1, 'description': 'Asserted after an FE when the data payload received between FS and FE contains errors'},
			'Fame synchronization error for VC15': {'access': 'read-write', 'bit_offset': 23, 'bit_width': 1, 'description': 'Asserted when an FE is not paired with a Frame Start (FS) on the same virtual channel'}
		}
	},
	'clock_lane_information': {'address_offset': 0x3C, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Clock Lane Information register', 'type': 'uint',
		'fields': {
			'stop_state': {'access': 'read-only', 'bit_offset': 1, 'bit_width': 1, 'description': 'Stop state on clock lane'}
		}
	},
	'lane_0_information': {'address_offset': 0x40, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Lane0 Information register', 'type': 'uint',
		'fields': {
			'SoT Sync error': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 1, 'description': 'Detection of SoT Synchronization Error'},
			'SoT error': {'access': 'read-only', 'bit_offset': 1, 'bit_width': 1, 'description': 'Detection of SoT Error'},
			'skewcalhs': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': 'Indicates the deskew reception'},
			'stop_state': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': 'etection of stop state'}
		}
	},
	'lane_1_information': {'address_offset': 0x44, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Lane1 Information register', 'type': 'uint',
		'fields': {
			'SoT Sync error': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 1, 'description': 'Detection of SoT Synchronization Error'},
			'SoT error': {'access': 'read-only', 'bit_offset': 1, 'bit_width': 1, 'description': 'Detection of SoT Error'},
			'skewcalhs': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': 'Indicates the deskew reception'},
			'stop_state': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': 'etection of stop state'}
		}
	},
	'lane_2_information': {'address_offset': 0x48, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Lane2 Information register', 'type': 'uint',
		'fields': {
			'SoT Sync error': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 1, 'description': 'Detection of SoT Synchronization Error'},
			'SoT error': {'access': 'read-only', 'bit_offset': 1, 'bit_width': 1, 'description': 'Detection of SoT Error'},
			'skewcalhs': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': 'Indicates the deskew reception'},
			'stop_state': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': 'etection of stop state'}
		}
	},
	'lane_3_information': {'address_offset': 0x4C, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'The Lane3 Information register', 'type': 'uint',
		'fields': {
			'SoT Sync error': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 1, 'description': 'Detection of SoT Synchronization Error'},
			'SoT error': {'access': 'read-only', 'bit_offset': 1, 'bit_width': 1, 'description': 'Detection of SoT Error'},
			'skewcalhs': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': 'Indicates the deskew reception'},
			'stop_state': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': 'etection of stop state'}
		}
	},
	'image_information_0': {'address_offset': 0x60, 'access': 'read-write;', 'size': 64, 'host_size': 8, 'description': 'Image Information 0 register', 'type': 'uint',
		'fields': {
			'byte_count': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 16, 'description': 'Byte count of current packet being processed by the control FSM'},
			'line_count': {'access': 'read-only', 'bit_offset': 16, 'bit_width': 16, 'description': 'Number of long packets written to line buffer'},
			'data_type': {'access': 'read-only', 'bit_offset': 32, 'bit_width': 6, 'description': 'Indicates the deskew reception'}
		}
	},
	'image_information_1': {'address_offset': 0x68, 'access': 'read-write;', 'size': 64, 'host_size': 8, 'description': 'Image Information 1 register', 'type': 'uint',
		'fields': {
			'byte_count': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 16, 'description': 'Byte count of current packet being processed by the control FSM'},
			'line_count': {'access': 'read-only', 'bit_offset': 16, 'bit_width': 16, 'description': 'Number of long packets written to line buffer'},
			'data_type': {'access': 'read-only', 'bit_offset': 32, 'bit_width': 6, 'description': 'Indicates the deskew reception'}
		}
	},
	'image_information_2': {'address_offset': 0x70, 'access': 'read-write;', 'size': 64, 'host_size': 8, 'description': 'Image Information 2 register', 'type': 'uint',
		'fields': {
			'byte_count': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 16, 'description': 'Byte count of current packet being processed by the control FSM'},
			'line_count': {'access': 'read-only', 'bit_offset': 16, 'bit_width': 16, 'description': 'Number of long packets written to line buffer'},
			'data_type': {'access': 'read-only', 'bit_offset': 32, 'bit_width': 6, 'description': 'Indicates the deskew reception'}
		}
	},
	'image_information_3': {'address_offset': 0x78, 'access': 'read-write;', 'size': 64, 'host_size': 8, 'description': 'Image Information 3 register', 'type': 'uint',
		'fields': {
			'byte_count': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 16, 'description': 'Byte count of current packet being processed by the control FSM'},
			'line_count': {'access': 'read-only', 'bit_offset': 16, 'bit_width': 16, 'description': 'Number of long packets written to line buffer'},
			'data_type': {'access': 'read-only', 'bit_offset': 32, 'bit_width': 6, 'description': 'Indicates the deskew reception'}
		}
	},

	'dphy_control': {'address_offset': 0x1000, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'Enable and soft reset control for PHY', 'type': 'uint',
		'fields': {
			'srst': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 1, 'description': 'Soft reset for D-PHY Controller'},
			'dphy_en': {'access': 'read-write', 'bit_offset': 1, 'bit_width': 1, 'description': 'D-PHY Enabled'}
		}
	},
	'dphy_idelay_tap': {'address_offset': 0x1004, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'Calibration of IDELAY in 7 series D-PHY RX configuration for lanes 1 to 4', 'type': 'uint',
		'fields': {
			'tap_lane0': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 5, 'description': 'Tap value for lane 0'},
			'tap_lane1': {'access': 'read-write', 'bit_offset': 8, 'bit_width': 5, 'description': 'Tap value for lane 1'},
			'tap_lane2': {'access': 'read-write', 'bit_offset': 16, 'bit_width': 5, 'description': 'Tap value for lane 2'},
			'tap_lane3': {'access': 'read-write', 'bit_offset': 24, 'bit_width': 5, 'description': 'Tap value for lane 3'}
		}
	},
	'dphy_init': {'address_offset': 0x1008, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'Initialization timer', 'type': 'uint'},
	'dphy_hs_timeout': {'address_offset': 0x1010, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'Watchdog timeout in high-speed mode', 'type': 'uint'},
	'dphy_esc_timeout': {'address_offset': 0x1014, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'ESC timeout', 'type': 'uint'},
	'dphy_cl_status': {'address_offset': 0x1018, 'access': 'read-only;', 'size': 32, 'host_size': 4, 'description': 'Status register for PHY error reporting for clock Lane', 'type': 'uint',
		'fields': {
			'mode': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 2, 'description': 'Mode'},
			'ulps': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': '(ULP State mode'},
			'init_done': {'access': 'read-only', 'bit_offset': 3, 'bit_width': 1, 'description': 'Set after the lane has completed initialization'},
			'stop_state': {'access': 'read-only', 'bit_offset': 4, 'bit_width': 1, 'description': 'Clock lane is in the Stop state'},
			'err_control': {'access': 'read-only', 'bit_offset': 5, 'bit_width': 1, 'description': 'Clock lane control error'}
		}
	},
	'dphy_dl0_status': {'address_offset': 0x101C, 'access': 'read-only;', 'size': 32, 'host_size': 4, 'description': 'Status register for PHY error reporting for clock Lane 0', 'type': 'uint',
		'fields': {
			'mode': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 2, 'description': 'Mode'},
			'ulps': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': '(ULP State mode'},
			'init_done': {'access': 'read-only', 'bit_offset': 3, 'bit_width': 1, 'description': 'Set after the lane has completed initialization'},
			'hs_abort': {'access': 'read-write', 'bit_offset': 4, 'bit_width': 1, 'description': 'Set after the Data Lane High-Speed Timeout'},
			'esc_abort': {'access': 'read-write', 'bit_offset': 5, 'bit_width': 1, 'description': 'This bit is set after the Data Lane Escape Timeout'},
			'stop_state': {'access': 'read-only', 'bit_offset': 6, 'bit_width': 1, 'description': 'Data lane is in the Stop state'},
			'pkt_count': {'access': 'read-only', 'bit_offset': 16, 'bit_width': 16, 'description': 'Number of packets received or transmitted on the data lane'}
		}
	},
	'dphy_dl1_status': {'address_offset': 0x1020, 'access': 'read-only;', 'size': 32, 'host_size': 4, 'description': 'Status register for PHY error reporting for clock Lane 1', 'type': 'uint',
		'fields': {
			'mode': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 2, 'description': 'Mode'},
			'ulps': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': '(ULP State mode'},
			'init_done': {'access': 'read-only', 'bit_offset': 3, 'bit_width': 1, 'description': 'Set after the lane has completed initialization'},
			'hs_abort': {'access': 'read-write', 'bit_offset': 4, 'bit_width': 1, 'description': 'Set after the Data Lane High-Speed Timeout'},
			'esc_abort': {'access': 'read-write', 'bit_offset': 5, 'bit_width': 1, 'description': 'This bit is set after the Data Lane Escape Timeout'},
			'stop_state': {'access': 'read-only', 'bit_offset': 6, 'bit_width': 1, 'description': 'Data lane is in the Stop state'},
			'pkt_count': {'access': 'read-only', 'bit_offset': 16, 'bit_width': 16, 'description': 'Number of packets received or transmitted on the data lane'}
		}
	},
	'dphy_dl2_status': {'address_offset': 0x1024, 'access': 'read-only;', 'size': 32, 'host_size': 4, 'description': 'Status register for PHY error reporting for clock Lane 2', 'type': 'uint',
		'fields': {
			'mode': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 2, 'description': 'Mode'},
			'ulps': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': '(ULP State mode'},
			'init_done': {'access': 'read-only', 'bit_offset': 3, 'bit_width': 1, 'description': 'Set after the lane has completed initialization'},
			'hs_abort': {'access': 'read-write', 'bit_offset': 4, 'bit_width': 1, 'description': 'Set after the Data Lane High-Speed Timeout'},
			'esc_abort': {'access': 'read-write', 'bit_offset': 5, 'bit_width': 1, 'description': 'This bit is set after the Data Lane Escape Timeout'},
			'stop_state': {'access': 'read-only', 'bit_offset': 6, 'bit_width': 1, 'description': 'Data lane is in the Stop state'},
			'pkt_count': {'access': 'read-only', 'bit_offset': 16, 'bit_width': 16, 'description': 'Number of packets received or transmitted on the data lane'}
		}
	},
	'dphy_dl3_status': {'address_offset': 0x1028, 'access': 'read-only;', 'size': 32, 'host_size': 4, 'description': 'Status register for PHY error reporting for clock Lane 3', 'type': 'uint',
		'fields': {
			'mode': {'access': 'read-only', 'bit_offset': 0, 'bit_width': 2, 'description': 'Mode'},
			'ulps': {'access': 'read-only', 'bit_offset': 2, 'bit_width': 1, 'description': '(ULP State mode'},
			'init_done': {'access': 'read-only', 'bit_offset': 3, 'bit_width': 1, 'description': 'Set after the lane has completed initialization'},
			'hs_abort': {'access': 'read-write', 'bit_offset': 4, 'bit_width': 1, 'description': 'Set after the Data Lane High-Speed Timeout'},
			'esc_abort': {'access': 'read-write', 'bit_offset': 5, 'bit_width': 1, 'description': 'This bit is set after the Data Lane Escape Timeout'},
			'stop_state': {'access': 'read-only', 'bit_offset': 6, 'bit_width': 1, 'description': 'Data lane is in the Stop state'},
			'pkt_count': {'access': 'read-only', 'bit_offset': 16, 'bit_width': 16, 'description': 'Number of packets received or transmitted on the data lane'}
		}
	},
	'dphy_hs_settle0': {'address_offset': 0x1030, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'HS_SETTLE timing control for lane 0', 'type': 'uint',
		'fields': {
			'hs_settle_ns': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 9, 'description': 'HS_SETTLE timing parameter'}
		}
	},
	'dphy_hs_settle1': {'address_offset': 0x1048, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'HS_SETTLE timing control for lane 1', 'type': 'uint',
		'fields': {
			'hs_settle_ns': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 9, 'description': 'HS_SETTLE timing parameter'}
		}
	},
	'dphy_hs_settle2': {'address_offset': 0x104C, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'HS_SETTLE timing control for lane 2', 'type': 'uint',
		'fields': {
			'hs_settle_ns': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 9, 'description': 'HS_SETTLE timing parameter'}
		}
	},
	'dphy_hs_settle3': {'address_offset': 0x1050, 'access': 'read-write;', 'size': 32, 'host_size': 4, 'description': 'HS_SETTLE timing control for lane 3', 'type': 'uint',
		'fields': {
			'hs_settle_ns': {'access': 'read-write', 'bit_offset': 0, 'bit_width': 9, 'description': 'HS_SETTLE timing parameter'}
		}
	}
}


class MipiRx(DefaultIP):
	"""Driver for MIPI CSI-2 Receiver Subsystem"""
	bindto = ['xilinx.com:ip:mipi_csi2_rx_subsystem:5.1']
	def __init__(self, description):
		description['registers'] = _registers
		super().__init__(description)

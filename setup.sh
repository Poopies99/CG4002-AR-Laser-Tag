#!/bin/bash
# Local Machine Port Number
port_num="8080"

# SoC Credentials
soc_server="stu.comp.nus.edu.sg"
soc_username="chris99"

# Xilinx Credentials
xilinx_server="192.168.95.221"
xilinx_username="xilinx"
xilinx_port_number="8080"

# Setup SSH Port Forward
ssh -f -N -L $port_num:$xilinx_server:$xilinx_port_number $soc_username@$soc_server
ssh -t -X $soc_username@$soc_server ssh -X $xilinx_username@$xilinx_server
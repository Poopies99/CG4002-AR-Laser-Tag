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
# Setup Reverse SSH between Local Machine and stu Server
ssh -f -N -R 5000:localhost:8012 chris99@stu.comp.nus.edu.sg
# Setup Reverse SSH between stu Server and ultra96
ssh -t $soc_username@$soc_server ssh -f -N -R 12345:localhost:5000 xilinx@192.168.95.221
echo "Hello"
#ssh -t -X $soc_username@$soc_server ssh -X $xilinx_username@$xilinx_server


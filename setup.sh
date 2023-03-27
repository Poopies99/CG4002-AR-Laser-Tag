#!/bin/bash
# Local Machine Port Number
port_num="8080"

# SoC Credentials
soc_server="stu.comp.nus.edu.sg"
soc_username="chris99"

# Xilinx Credentials
XILINX_SERVER="192.168.95.221"
xilinx_username="xilinx"
XILINX_PORT_NUMber="8080"

# Setup SSH Port Forward
ssh -f -N -L $port_num:$XILINX_SERVER:$XILINX_PORT_NUMber $soc_username@$soc_server
# Setup Reverse SSH between Local Machine and stu Server
#ssh -f -N -R 5050:localhost:5050 chris99@stu.comp.nus.edu.sg
## Setup Reverse SSH between stu Server and ultra96
#ssh -t -N $soc_username@$soc_server ssh -N -R 5050:localhost:5050 xilinx@192.168.95.221
ssh -t -X $soc_username@$soc_server ssh -X $xilinx_username@$XILINX_SERVER


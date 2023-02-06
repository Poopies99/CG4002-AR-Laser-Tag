#!/bin/bash

# SoC Credentials
soc_server="stu.comp.nus.edu.sg"
soc_username="chris99"

# Xilinx Credentials
xilinx_server="192.168.95.221"
xilinx_username="xilinx"

# Remote Port Forward
ssh -t -X $soc_username@$soc_server ssh -X $xilinx_username@$xilinx_server "python3 test.py"
#ssh $soc_username@$soc_server "python3 test.py"
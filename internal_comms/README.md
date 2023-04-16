# CG4002 Computer Engineering Capstone Project
### Group B14 Internal Communications

This is part of a project to implement a 2-player laser tag system. 
The scripts in this repository provide an API for the internal communications between the Bluno Beetles and a relay laptop.

This project uses the bluepy library by Ian Harvey as a Python interface for Bluetooth LE on Linux.
The source code can be found [here](https://github.com/IanHarvey/bluepy).

At present, the scripts only run on Linux.
The scripts have been tested to work on Ubuntu Desktop 22.04 LTS with Python 3.10. 
(Making use of Ubuntu CLI or a VM has proven to not work for this particular implementation.)

## Setup
1. Clone the repository.
2. Download the bluepy library by following the installation instructions provided [here](https://github.com/IanHarvey/bluepy).
3. Download the `multipledispatch` library using the following command.

```
pip3 install multipledispatch
```

4. Change the "FILEPATH" variable within `main.py` to the directory containing this repository.
5. Update the MAC addresses of the Bluno Beetles that are being used. 
This can be done by adding in the corresponding MAC addresses to the following lines in `helper/constant.py`

```
# MAC addresses of Bluno Beetles
# Player 1
P1_IR_TRANSMITTER = ""
P1_IR_RECEIVER = ""
P1_IMU_SENSOR = ""

# Player 2
P2_IR_TRANSMITTER = ""
P2_IR_RECEIVER = ""
P2_IMU_SENSOR = ""
```

## Running the program

Run the program using the following command.

```
sudo python3 main.py
```

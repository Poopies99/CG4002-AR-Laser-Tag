# CG4002 Computer Engineering Capstone Project
### Group B14 External Communications

The computer engineering capstone consists of a laser tag game where 2 players are allowed to play at any point in time.

## Systems Architecture
![alt text](https://user-images.githubusercontent.com/69495787/232277660-a2485a2f-5e9a-498f-8b00-76c2ac73e9fb.jpg)

## External Communications
After cloning the repository run
``
./setup.sh
``
 to establish SSH tunnel to the Ultra96 via the stu server and clone the repository in the Ultra96.

Pip install numpy and pandas for AI thread
```
pip3 install numpy
pip3 install pandas
```

Then run the following command to start the Ultra96 server
```
cd <repository name>
python3 ultra.py
```
import pandas as pd
from intcomm import IntComm
import time

if __name__ == "__main__":

    # change this according to serial port
    # 0: "/dev/ttyACM0"
    # 1: "/dev/ttyACM1"
    # 2: "/dev/ttyACM2"
    intcomm = IntComm(0) # confirm comms port
    all_data = []
    print("Start")
    try:
        while True:
            # collecting data upon key press and 1s sleep timer
            input("Press any key to start data collection...")
            time.sleep(1)

            start_time = time.time()
            print("Recording for 1 second...")

            # assuming all actions within 1 second of key press
            while time.time() - start_time < 1:
                data = intcomm.get_line()
                print(data)
                if len(data) == 0 or data[0] != "#":
                    print("Invalid data:", data)
                    continue

                data = data[1:].split(",")
                if len(data) == 8:
                    yaw, pitch, roll, accx, accy, accz, flex1, flex2 = data
                    
                    all_data.append(
                        [yaw, pitch, roll, accx, accy, accz, flex1, flex2]
                    )

    except KeyboardInterrupt:
        print("terminating program")
    except Exception:
        print("an error occured")

    df = pd.DataFrame(all_data)
    print(df.head())
    df.columns = ["yaw", "pitch", "roll", "ax", "ay", "az", "flex1", "flex2"]
    df.to_csv("training_data.csv", sep=",")
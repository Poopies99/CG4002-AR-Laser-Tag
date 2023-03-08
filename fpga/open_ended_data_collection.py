import time
import pandas as pd
import numpy as np
import csv

if __name__ == "__main__":
    raw_headers = ["flex1", "flex2", "yaw", "pitch", "roll", "accX", "accY", "accZ"]

    all_data = []

    def record_data():
        data = ... # TODO comms - refactor to call in data
        if len(data) == 8:
            flex1, flex2, yaw, pitch, roll, accX, accY, accZ = data

            all_data.append([flex1, flex2, yaw, pitch, roll, accX, accY, accZ])
        
    def start_recording():
        print("Recording for 10 seconds...\n")
        start_time = time.time()
        while time.time() - start_time < 10:
            record_data()

    def collect_data():
        input("Press any key to start data collection...\n")
        start_time = time.time()
        print("Collecting data...")
        while True:
            record_data()
            if time.time() - start_time >= 10:
                break

        # Convert data to DataFrame
        df = pd.DataFrame(all_data, columns=raw_headers)
        print(df[['yaw', 'pitch', 'roll', 'accX', 'accY', 'accZ']].head(40))
        print(f"Number of rows and columns: {df.shape[0]} by {df.shape[1]}")
        user_input = input("Does the data look ok? (y/n): ")

        if user_input.lower() == "y":
            # Store raw data into a new CSV file
            filename = time.strftime("%Y%m%d-%H%M%S") + "_raw_open_ended.csv"
            df.to_csv(filename, index=False, header=True)

            # Clear raw data list
            all_data.clear()
            print("Data processed and saved to CSV file.")
        else:
            print("Data not processed.")

    collect_data()

import threading
import numpy as np
from scipy import stats
import pandas as pd
import time
import json
from pynq import Overlay


class Process:
    super().__init__()

    @staticmethod
    def preprocess_data(self, data):
        mean = np.mean(data)
        std = np.std(data)
        variance = np.var(data)
        range = np.max(data) - np.min(data)
        peak_to_peak_amplitude = np.abs(np.max(data) - np.min(data))
        mad = np.median(np.abs(data - np.median(data)))
        root_mean_square = np.sqrt(np.mean(np.square(data)))
        interquartile_range = stats.iqr(data)
        percentile_75 = np.percentile(data, 75)
        energy = np.sum(data ** 2)

        output_array = np.empty((1, 10))
        output_array[0] = [mean, std, variance, range, peak_to_peak_amplitude, mad, root_mean_square,
                           interquartile_range, percentile_75, energy]

        return output_array


    def preprocessing_and_mlp(self, arr):
        processed_data = []

        # Set the window size for the median filter
        window_size = 7

        df = pd.DataFrame(arr)
        df_filtered = df.rolling(window_size, min_periods=1, center=True).median()

        arr = df_filtered.values

        # Split the rows into 8 groups
        group_size = 5
        num_groups = 8

        # Loop through each group and column, and compute features
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            group = arr[start_idx:end_idx, :]

            group_data = []
            for column in range(arr.shape[1]):
                column_data = group[:, column]
                column_data = column_data.reshape(1, -1)

                temp_processed = self.preprocess_data(column_data)
                temp_processed = temp_processed.flatten()

                group_data.append(temp_processed)

            processed_data.append(np.concatenate(group_data))

        processed_data_arr = np.concatenate(processed_data)

        predicted_label = self.MLP_Driver(processed_data_arr)

        return predicted_label

    def MLP_Overlay(self, data):
        start_time = time.time()

        # reshape data to match in_buffer shape
        data = np.reshape(data, (35,))

        self.in_buffer[:] = data

        self.dma.sendchannel.transfer(self.in_buffer)
        self.dma.recvchannel.transfer(self.out_buffer)

        # wait for transfer to finish
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # print output buffer
        print("mlp done with output: " + " ".join(str(x) for x in self.out_buffer))

        print(f"MLP time taken so far output: {time.time() - start_time}")

        return self.out_buffer

    def MLP_Driver(self, data):
        # MLP Library
        # mlp = joblib.load('mlp_model.joblib') # localhost
        # mlp = joblib.load('/home/xilinx/mlp_model.joblib') # board

        # sample data for sanity check
        # test_input = np.array([0.1, 0.2, 0.3, 0.4] * 120).reshape(1, -1)

        # Scaler
        test_input_rescaled = (data - self.mean) / np.sqrt(self.variance)  # TODO - use this for real data
        # test_input_rescaled = (test_input - self.mean) / np.sqrt(self.variance)
        # print(f"test_input_rescaled: {test_input_rescaled}\n")

        # PCA
        test_input_math_pca = np.dot(test_input_rescaled, self.pca_eigvecs_transposed)
        # print(f"test_input_math_pca: {test_input_math_pca}\n")

        # MLP - TODO PYNQ Overlay
        predicted_labels = self.MLP_Overlay(test_input_math_pca)  # return 1x4 softmax array
        print(f"MLP pynq overlay predicted: {predicted_labels} \n")
        np_output = np.array(predicted_labels)
        largest_index = np_output.argmax()

        predicted_label = self.action_map[largest_index]

        # print largest index and largest action of MLP output
        print(f"largest index: {largest_index} \n")
        print(f"MLP overlay predicted: {predicted_label} \n")

        # MLP - LIB Overlay
        # predicted_label = mlp.predict(test_input_math_pca.reshape(1, -1))
        # print(f"MLP lib overlay predicted: {predicted_label} \n")

        return predicted_label


class AIModel(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.shutdown = threading.Event()

        self.columns = ['gx', 'gy', 'gz', 'accX', 'accY', 'accZ']

        self.factors = ['mean', 'std', 'variance', 'range', 'peak_to_peak_amplitude',
                        'mad', 'root_mean_square', 'interquartile_range', 'percentile_75',
                        'energy']

        self.num_groups = 8
        self.headers = [f'grp_{i + 1}_{column}_{factor}' for i in range(self.num_groups)
                        for column in self.columns for factor in self.factors]

        self.headers.extend(['action'])

        # defining game action dictionary
        self.action_map = {0: 'G', 1: 'L', 2: 'R', 3: 'S'}

        # load PCA model
        # read the contents of the arrays.txt file
        with open("dependencies/arrays.txt", "r") as f:
            data = json.load(f)

        # extract the weights and bias arrays
        self.scaling_factor = data['scaling_factor']
        self.mean = data['mean']
        self.variance = data['variance']
        self.pca_eigvecs_list = data['pca_eigvecs_list']

        self.pca_eigvecs_transposed = [list(row) for row in zip(*self.pca_eigvecs_list)]

        # PYNQ overlay
        self.overlay = Overlay("pca_mlp_1.bit")
        self.dma = self.overlay.axi_dma_0

        # Allocate input and output buffers once
        self.in_buffer = pynq.allocate(shape=(35,), dtype=np.float32)
        self.out_buffer = pynq.allocate(shape=(4,), dtype=np.float32)

        self.detection_time = DetectionTime()

    def sleep(self, seconds):
        start_time = time.time()
        while time.time() - start_time < seconds:
            pass

    def generate_simulated_data(self):
        gx = random.uniform(-9, 9)  # TODO - assumption: gyro x,y,z change btwn -9 to 9
        gy = random.uniform(-9, 9)
        gz = random.uniform(-9, 9)
        accX = random.uniform(-9, 9)
        accY = random.uniform(-9, 9)
        accZ = random.uniform(-9, 9)
        return [gx, gy, gz, accX, accY, accZ]

    def preprocess_data(self, data):
        # standard data processing techniques
        mean = np.mean(data)
        std = np.std(data)
        variance = np.var(data)
        range = np.max(data) - np.min(data)
        peak_to_peak_amplitude = np.abs(np.max(data) - np.min(data))
        mad = np.median(np.abs(data - np.median(data)))
        root_mean_square = np.sqrt(np.mean(np.square(data)))
        interquartile_range = stats.iqr(data)
        percentile_75 = np.percentile(data, 75)
        energy = np.sum(data ** 2)

        output_array = np.empty((1, 10))
        output_array[0] = [mean, std, variance, range, peak_to_peak_amplitude, mad, root_mean_square,
                           interquartile_range, percentile_75, energy]

        return output_array

    def preprocessing_and_mlp(self, arr):
        processed_data = []

        # Set the window size for the median filter
        window_size = 7

        df = pd.DataFrame(arr)
        df_filtered = df.rolling(window_size, min_periods=1, center=True).median()

        arr = df_filtered.values

        # Split the rows into 8 groups
        group_size = 5
        num_groups = 8

        # Loop through each group and column, and compute features
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            group = arr[start_idx:end_idx, :]

            group_data = []
            for column in range(arr.shape[1]):
                column_data = group[:, column]
                column_data = column_data.reshape(1, -1)

                temp_processed = self.preprocess_data(column_data)
                temp_processed = temp_processed.flatten()

                group_data.append(temp_processed)

            processed_data.append(np.concatenate(group_data))

        processed_data_arr = np.concatenate(processed_data)

        predicted_label = self.MLP_Driver(processed_data_arr)

        return predicted_label

    def MLP_Overlay(self, data):
        start_time = time.time()

        # reshape data to match in_buffer shape
        data = np.reshape(data, (35,))

        self.in_buffer[:] = data

        self.dma.sendchannel.transfer(self.in_buffer)
        self.dma.recvchannel.transfer(self.out_buffer)

        # wait for transfer to finish
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # print output buffer
        print("mlp done with output: " + " ".join(str(x) for x in self.out_buffer))

        print(f"MLP time taken so far output: {time.time() - start_time}")

        return self.out_buffer

    def MLP_Driver(self, data):
        # MLP Library
        # mlp = joblib.load('mlp_model.joblib') # localhost
        # mlp = joblib.load('/home/xilinx/mlp_model.joblib') # board

        # sample data for sanity check
        # test_input = np.array([0.1, 0.2, 0.3, 0.4] * 120).reshape(1, -1)

        # Scaler
        test_input_rescaled = (data - self.mean) / np.sqrt(self.variance)  # TODO - use this for real data
        # test_input_rescaled = (test_input - self.mean) / np.sqrt(self.variance)
        # print(f"test_input_rescaled: {test_input_rescaled}\n")

        # PCA
        test_input_math_pca = np.dot(test_input_rescaled, self.pca_eigvecs_transposed)
        # print(f"test_input_math_pca: {test_input_math_pca}\n")

        # MLP - TODO PYNQ Overlay
        predicted_labels = self.MLP_Overlay(test_input_math_pca)  # return 1x4 softmax array
        print(f"MLP pynq overlay predicted: {predicted_labels} \n")
        np_output = np.array(predicted_labels)
        largest_index = np_output.argmax()

        predicted_label = self.action_map[largest_index]

        # print largest index and largest action of MLP output
        print(f"largest index: {largest_index} \n")
        print(f"MLP overlay predicted: {predicted_label} \n")

        # MLP - LIB Overlay
        # predicted_label = mlp.predict(test_input_math_pca.reshape(1, -1))
        # print(f"MLP lib overlay predicted: {predicted_label} \n")

        return predicted_label

    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        K = float(input("threshold value? "))

        # Initialize arrays to hold the current and previous data packets
        current_packet = np.zeros((6, 6))
        previous_packet = np.zeros((6, 6))
        data_packet = np.zeros((40, 6))
        is_movement_counter = 0
        movement_watchdog = False
        loop_count = 0

        # Enter the main loop
        while True:
            # runs loop 6 times and packs the data into groups of 6
            if ai_queue:
                q_data = ai_queue.get()
                ai_queue.task_done()

                new_data = np.array(q_data)
                new_data[-3:] = [x / 100.0 for x in new_data[-3:]]

                # print(" ".join([f"{x:.3f}" for x in new_data]))
                timestamp = time.time()
                tz = datetime.timezone(datetime.timedelta(hours=8))  # UTC+8
                dt_object = datetime.datetime.fromtimestamp(timestamp, tz)
                # print(f"- packet received at {dt_object} \n")

                # Pack the data into groups of 6
                current_packet[loop_count] = new_data

                # Update loop_count
                loop_count = (loop_count + 1) % 5

                if loop_count % 5 == 0:

                    curr_mag = np.sum(np.square(np.mean(current_packet[:, -3:], axis=1)))
                    prev_mag = np.sum(np.square(np.mean(previous_packet[:, -3:], axis=1)))

                    # Check for movement detection
                    if not movement_watchdog and curr_mag - prev_mag > K:
                        self.detection_time.start_timer()
                        print("Movement detected!")
                        # print currr and prev mag for sanity check
                        print(f"curr_mag: {curr_mag} \n")
                        print(f"prev_mag: {prev_mag} \n")
                        is_movement_counter = 1
                        movement_watchdog = True
                        # append previous and current packet to data packet
                        data_packet = np.concatenate((previous_packet, current_packet), axis=0)

                    # movement_watchdog activated, count is_movement_counter from 0 up 6 and append current packet each time
                    if movement_watchdog:
                        if is_movement_counter <= 6:
                            data_packet = np.concatenate((data_packet, current_packet), axis=0)
                            is_movement_counter += 1
                        else:
                            # print dimensions of data packet
                            # print(f"data_packet dimensions: {data_packet.shape} \n")
                            # display_df = pd.DataFrame(data_packet, columns=self.columns)
                            # print(display_df.head(40))

                            # If we've seen 6 packets since the last movement detection, preprocess and classify the data
                            predicted_label = self.preprocessing_and_mlp(data_packet)
                            print(f"output from MLP in main: \n {predicted_label} \n")  # print output of MLP
                            self.detection_time.end_timer()
                            # movement_watchdog deactivated, reset is_movement_counter
                            movement_watchdog = False
                            is_movement_counter = 0
                            # reset arrays to zeros
                            current_packet = np.zeros((6, 6))
                            previous_packet = np.zeros((6, 6))
                            data_packet = np.zeros((40, 6))

                    # Update the previous packet
                    previous_packet = current_packet.copy()
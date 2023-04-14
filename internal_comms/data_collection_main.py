import sys

# File path of project directory
# TODO change file path to ur current pwd
FILEPATH = '/home/kenneth/Desktop/CG4002/cg4002-internal-comms/'

# importing necessary module directories
sys.path.append(FILEPATH + 'bluno_beetle')
sys.path.append(FILEPATH + 'helper')

from bluno_beetle import BlunoBeetle
from bluno_beetle_udp import BlunoBeetleUDP
from _socket import SHUT_RDWR
from queue import Queue
import random
import csv
import constant
import socket
import threading
import traceback
import time
from ble_packet import BLEPacket
from collections import deque

class Training(threading.Thread):
    collect_flag = False
    training_queue = deque()

    def __init__(self):
        super().__init__()

        self.packer = BLEPacket()

        self.shutdown = threading.Event()

        self.columns = ['gx', 'gy', 'gz', 'accX', 'accY', 'accZ']
        self.empty_line = []

        self.factors = ['mean', 'std', 'variance', 'min', 'max', 'range', 'peak_to_peak_amplitude',
                        'mad', 'root_mean_square', 'interquartile_range', 'percentile_75',
                        'skewness', 'kurtosis', 'zero_crossing_rate', 'energy']

        self.headers = [f'{raw_header}_{factor}' for raw_header in self.columns for factor in self.factors]
        self.headers.extend(['action', 'timestamp'])

        self.action_map = {0: 'GRENADE', 1: 'LOGOUT', 2: 'SHIELD', 3: 'RELOAD'}
        self.filename = FILEPATH + "training/new_logout.csv"
        
        self.dataset_count = 0
    def sleep(self, seconds):
        start_time = time.time()
        while time.time() - start_time < seconds:
            pass

    # def preprocess_data(self, data):
    #     mean = np.mean(data)
    #     std = np.std(data)
    #     variance = np.var(data)
    #     min = np.min(data)
    #     max = np.max(data)
    #     range = np.max(data) - np.min(data)
    #     peak_to_peak_amplitude = np.abs(np.max(data) - np.min(data))
    #     mad = np.median(np.abs(data - np.median(data)))
    #     root_mean_square = np.sqrt(np.mean(np.square(data)))
    #     interquartile_range = stats.iqr(data)
    #     percentile_75 = np.percentile(data, 75)
    #     skewness = stats.skew(data.reshape(-1, 1))[0]
    #     kurtosis = stats.kurtosis(data.reshape(-1, 1))[0]
    #     zero_crossing_rate = ((data[:-1] * data[1:]) < 0).sum()
    #     energy = np.sum(data ** 2)
    #     # entropy = stats.entropy(data, base=2)
    #
    #     output_array = [mean, std, variance, min, max, range, peak_to_peak_amplitude,
    #                     mad, root_mean_square, interquartile_range, percentile_75,
    #                     skewness, kurtosis, zero_crossing_rate, energy]
    #
    #     output_array = np.array(output_array)
    #     return output_array.reshape(1, -1)

    # def preprocess_dataset(self, df):
    #     processed_data = []
    #
    #     # Loop through each column and compute features
    #     for column in df.columns:
    #         column_data = df[column].values
    #         column_data = column_data.reshape(1, -1)
    #         # print column1 values
    #         print(f"column_data: {column_data}\n")
    #         print("Data type of column_data:", type(column_data))
    #         print("Size of column_data:", column_data.size)
    #
    #         temp_processed = self.preprocess_data(column_data)
    #
    #         # print(processed_column_data)
    #         # Append processed column data to main processed data array
    #         processed_data.append(temp_processed)
    #
    #     processed_data_arr = np.concatenate(processed_data)
    #
    #     # reshape into a temporary dataframe of 8x14
    #     temp_df = pd.DataFrame(processed_data_arr.reshape(8, -1), index=self.columns, columns=self.factors)
    #
    #     # print the temporary dataframe
    #     print(f"processed_data: \n {temp_df} \n")
    #     print(f"len processed_data: {len(processed_data_arr)}\n")
    #
    #     return processed_data_arr

    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        #with open(self.filename, "a") as f:
        #    writer = csv.writer(f)
        #    writer.writerow(self.columns)

        all_data = []

        i = 0
        while not self.shutdown.is_set():
            try:
                input("start?")
                all_data = []
                #Training.collect_flag = True
                #global global_flag
                #global_flag = True
                Training.training_queue.clear()
                print(len(Training.training_queue))
                # while i < 41:
                #     data = training_queue.get()
                #     self.packer.unpack(data)
                #     data = self.packer.get_flex_data() + self.packer.get_euler_data() +self.packer.get_acc_data()
                #     print(f"data: {data} \n")
                #
                #     if len(data) == 0:
                #         print("Invalid data:", data)
                #         continue
                #     if len(data) == 8:
                #         flex1, flex2, gx, gy, gz, accX, accY, accZ = data
                #         all_data.append([flex1, flex2, gx, gy, gz, accX / 100, accY / 100, accZ / 100])
                start_time = time.time()

                while i < 40:
                    # getting data - simulation
                    # data = self.generate_simulated_data()
                    # print(f"data: {data} \n")

                    # # getting data - actl
                    if not Training.training_queue:
                        continue

                    data = Training.training_queue.popleft()
                    self.packer.unpack(data)
                    data = self.packer.get_euler_data() + self.packer.get_acc_data()
                    print(f"data: {data} \n")

                    if len(data) == 0:
                        print("Invalid data:", data)
                        continue
                    if len(data) == 6:
                        gx, gy, gz, accX, accY, accZ = data
                        all_data += [gx, gy, gz, accX, accY, accZ]

                    self.sleep(0.05)
                    i += 1
                #Training.collect_flag = False
                #global_flag = False
                size = len(all_data)
                print(size)
                if size >= 240:
                    extra = size - 240
                    all_data = all_data[extra:]
                else:
                    lack = 240 - size
                    padding = []
                    for i in range(lack):
                        padding.append(1)
                    all_data = padding + all_data

                print(len(all_data))
                i = 0
                #for i in all_data:
                #    print(i)
                # creating df for prneview
                # df = pd.DataFrame(all_data, columns=self.columns)
                # # creating res to output differences
                # res = pd.DataFrame(columns=self.columns)
                #
                # for j in range(len(df)):
                #     diff = df.iloc[j] - df.iloc[j - 1]
                #     res = res.append(diff, ignore_index=True)

                # Show user the data and prompt for confirmation
                # print(res[['gx', 'gy', 'gz', 'accX', 'accY', 'accZ']].head(40))
                # print(f"Number of rows and columns: {df.shape[0]} by {df.shape[1]}")

                ui = input("data ok? y/n")
                if ui.lower() == "y":

                    # time_now = time.strftime("%Y%m%d-%H%M%S")
                    #
                    # res_arr = res.values.reshape(1, -1)
                    # res_arr = np.append(res_arr, time_now)

                    # Store data into a new CSV file
                    
                    with open(self.filename, "a") as f:
                        writer = csv.writer(f)
                        #writer.writerow(self.empty_line)

                        #for row in all_data:
                        #    writer.writerow(row)
                        writer.writerow(all_data)
                        writer.writerow(self.empty_line)

                    # # Clear raw data list
                    # all_data = []
                    # res_arr = []
                    # i = 0
                    #
                    # # Preprocess data
                    # processed_data = self.preprocess_dataset(res)
                    #
                    # # Prompt user for label
                    # label = input("Enter label (G = GRENADE, R = RELOAD, S = SHIELD, L = LOGOUT): ")
                    #
                    # # Append label, timestamp to processed data
                    # processed_data = np.append(processed_data, label)
                    # processed_data = np.append(processed_data, time_now)
                    #
                    # # Append processed data to CSV file
                    # with open("/home/kenneth/CG4002/training/processed_data.csv", "a") as f:
                    #     writer = csv.writer(f)
                    #     # writer.writerow(self.headers)
                    #     writer.writerow(processed_data)
                    self.dataset_count += 1
                    print("Data {} processed and saved to CSV file.".format(self.dataset_count))
                else:
                    res_arr = []
                    # i = 0
                    print("not proceed, restart")
            except Exception as _:
                traceback.print_exc()
                self.close_connection()
                print("an error occurred")


class Controller(threading.Thread):
    def __init__(self, params):
        super().__init__()

        # Create a TCP/IP socket
        # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # self.client_socket = client_socket
        # self.connection = client_socket.connect(("localhost", 8080))

        # Flags
        self.shutdown = threading.Event()
        
        self.beetles = [
                #BlunoBeetle(params[0]), 
                #BlunoBeetle(params[1]), 
                BlunoBeetleUDP(params[0])
            ]
        
        # For statistics calculation
        self.start_time = 0
        self.prev_time = 0
        self.prev_processed_bit_count = 0
        self.current_data_rate = 0

    def close_connection(self):
        # self.connection.shutdown(SHUT_RDWR)
        # self.connection.close()
        self.shutdown.set()
        # self.client_socket.close()

        print("Shutting Down Connection")
  
    def run_threads(self):
        # create thread for printing statistics
        print_thread = threading.Thread(target=self.print_statistics, args=())

        for i in range(13):
            print()

        self.start_time = time.perf_counter()

        for beetle in self.beetles:
            beetle.start()

        #print_thread.start()
    
    # run() function invoked by thread.start()
    def run(self):
        self.run_threads()

        while not self.shutdown.is_set():
            try:
                # global global_flag
                #message = input("Enter message to be sent: ")
                #if message == 'q':
                #    break
                data = BlunoBeetle.packet_queue.get()
                #training_queue.put(data)
                #if Training.collect_flag == True:
                Training.training_queue.append(data)
                #print(data)
                # self.client_socket.send(data)
            except Exception as _:
                # traceback.print_exc()
                self.close_connection()
    
    # prints beetle data and statistics to std output
    def print_statistics(self):
        while True:
            for i in range(13):
                print(constant.LINE_UP, end="")

            print("***********************************************************************************************************")
            processed_bit_count = 0
            fragmented_packet_count = 0
            for beetle in self.beetles:
                processed_bit_count += beetle.get_processed_bit_count()
                fragmented_packet_count += beetle.get_fragmented_packet_count()
                beetle.print_beetle_info()

            print("Statistics".ljust(80))
            current_time = time.perf_counter()
            if current_time - self.prev_time >= 1:
                self.current_data_rate = ((processed_bit_count - self.prev_processed_bit_count) / 1000) / (current_time - self.prev_time)
                self.prev_time = current_time
                self.prev_processed_bit_count = processed_bit_count
            print("Current data rate: {} kbps".ljust(80).format(self.current_data_rate))
            print("Average Data rate: {} kbps".ljust(80).format(
                (processed_bit_count / 1000) / (current_time - self.start_time)
            ))
            print("No. of fragmented packets: {}".ljust(80).format(fragmented_packet_count))
            print("************************************************************************************************************")


if __name__ == '__main__':
    controller = Controller([
        #(1, constant.P1_IR_TRANSMITTER),    # P1 gun (IR transmitter)
        #(2, constant.P1_IR_RECEIVER),       # P1 vest (IR receiver)
        [0, 3, constant.P1_IMU_SENSOR],        # P1 glove (IMU and flex sensors)
        #(1, constant.P2_IR_TRANSMITTER),    # P2 gun (IR transmitter)
        #(2, constant.P2_IR_RECEIVER),       # P2 vest (IR receiver)
        #[1, 6, constant.P2_IMU_SENSOR]         # P2 glove (IMU and flex sensors)
    ])
    controller.start()

    print('Starting Training Thread')
    train = Training()
    train.start()


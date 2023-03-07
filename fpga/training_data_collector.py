import pandas as pd
import time
import numpy as np
from scipy import stats, signal
import csv
import os

def preprocess_data(df):
    
    # Compute features for each column
    def compute_mean(data):
        return np.mean(data)

    def compute_variance(data):
        return np.var(data)

    def compute_median(data):
        return np.median(data)

    def compute_root_mean_square(data):
        return np.sqrt(np.mean(np.square(data)))

    def compute_interquartile_range(data):
        return stats.iqr(data)

    def compute_percentile_75(data):
        return np.percentile(data, 75)

    def compute_kurtosis(data):
        return stats.kurtosis(data)

    def compute_min_max(data):
        return np.max(data) - np.min(data)

    def compute_signal_magnitude_area(data):
        return np.sum(data) / len(data)

    def compute_zero_crossing_rate(data):
        return ((data[:-1] * data[1:]) < 0).sum()

    def compute_spectral_centroid(data):
        spectrum = np.abs(np.fft.rfft(data))
        normalized_spectrum = spectrum / np.sum(spectrum)
        normalized_frequencies = np.linspace(0, 1, len(spectrum))
        spectral_centroid = np.sum(normalized_frequencies * normalized_spectrum)
        return spectral_centroid

    def compute_spectral_entropy(data):
        freqs, power_density = signal.welch(data)
        return stats.entropy(power_density)

    def compute_spectral_energy(data):
        freqs, power_density = signal.welch(data)
        return np.sum(np.square(power_density))

    def compute_principle_frequency(data):
        freqs, power_density = signal.welch(data)
        return freqs[np.argmax(np.square(power_density))]
    
    processed_data = []

    # Loop through each column and compute features
    for column in df.columns:
        column_data = df[column]

        # Compute features for the column
        mean = compute_mean(column_data)
        variance = compute_variance(column_data)
        median = compute_median(column_data)
        root_mean_square = compute_root_mean_square(column_data)
        interquartile_range = compute_interquartile_range(column_data)
        percentile_75 = compute_percentile_75(column_data)
        kurtosis = compute_kurtosis(column_data)
        min_max = compute_min_max(column_data)
        signal_magnitude_area = compute_signal_magnitude_area(column_data)
        zero_crossing_rate = compute_zero_crossing_rate(column_data)
        spectral_centroid = compute_spectral_centroid(column_data)
        spectral_entropy = compute_spectral_entropy(column_data)
        spectral_energy = compute_spectral_energy(column_data)
        principle_frequency = compute_principle_frequency(column_data)

        # Store features in list
        processed_column_data = [mean, variance, median, root_mean_square, 
                                interquartile_range, percentile_75, kurtosis, min_max, 
                                signal_magnitude_area, zero_crossing_rate, spectral_centroid, 
                                spectral_entropy, spectral_energy, principle_frequency]
        print(processed_column_data)
        # Append processed column data to main processed data array
        processed_data.append(processed_column_data)

    processed_data_arr = np.concatenate(processed_data)

    return processed_data_arr


if __name__ == "__main__":
    all_data = []

    raw_headers = ['flex1', 'flex2', 'yaw', 'pitch', 'roll', 'accX', 'accY', 'accZ']

    # defining headers for post processing
    factors = ['mean', 'variance', 'median', 'root_mean_square', 'interquartile_range',            
            'percentile_75', 'kurtosis', 'min_max', 'signal_magnitude_area', 'zero_crossing_rate',            
            'spectral_centroid', 'spectral_entropy', 'spectral_energy', 'principle_frequency']

    headers = [f'{raw_header}_{factor}' for raw_header in raw_headers for factor in factors]
    headers.append('action')

    print("Start")
    if 1 == 1:
        while True:
            # collecting data upon key press and 1s sleep timer
            input("Press any key to start data collection...\n")

            start_time = time.time()
            print("Recording for 2 seconds...\n")

            # assuming all actions within 1 second of key press
            while time.time() - start_time < 2:
                data = ... # TODO comms - refactor to call in data
                print(f"data: {data} \n")
                # if len(data) == 0 or data[0] != "#":
                #     print("Invalid data:", data)
                #     continue

                # data = data[1:].split(",")
                if len(data) == 8:
                    flex1, flex2, yaw, pitch, roll, accX, accY, accZ = data
                    
                    all_data.append(
                        [flex1, flex2, yaw, pitch, roll, accX, accY, accZ]
                    )

            # Convert data to DataFrame
            df = pd.DataFrame([all_data], columns=raw_headers)

            # Show user the data and prompt for confirmation
            print(df[['yaw', 'pitch', 'roll', 'accX', 'accY', 'accZ']].head(40))
            print(f"Number of rows and columns: {df.shape[0]} by {df.shape[1]}")

            user_input = input("Does the data look ok? (y/n): ")
            if user_input.lower() == "y":

                # Store raw data into a new CSV file
                filename = time.strftime("%Y%m%d-%H%M%S") + "_raw.csv"
                df.to_csv(filename, index=False, header=True)
                
                # process data
                processed_data = preprocess_data(df)
                print("processed_data: \n")
                print(processed_data)
                print("\n")

                # Prompt user for label
                label = input("Enter label (G = GRENADE, R = RELOAD, S = SHIELD, L = LOGOUT): ")

                # Append label to processed data
                processed_data = np.append(processed_data, label)

                # Append processed data to CSV file
                with open("processed_data.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers) # can remove aft file is init 
                    writer.writerow(processed_data)

                # Clear raw data list
                raw_data = []
                print("Data processed and saved to CSV file.")
            else:
                print("Data not processed.")



from ble_packet import BLEPacket
import csv
import pandas as pd
import numpy as np
import scipy
from scipy import stats, signal


def preprocess_data(df):
    def compute_mean(data):
        return np.mean(data)

    def compute_variance(data):
        return np.var(data)

    def compute_median_absolute_deviation(data):
        return stats.median_abs_deviation(data, axis=None)

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
        median_absolute_deviation = compute_median_absolute_deviation(column_data)
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
        processed_column_data = [mean, variance, median_absolute_deviation, root_mean_square,
                                 interquartile_range, percentile_75, kurtosis, min_max,
                                 signal_magnitude_area, zero_crossing_rate, spectral_centroid,
                                 spectral_entropy, spectral_energy, principle_frequency]
        print(processed_column_data)
        # Append processed column data to main processed data array
        processed_data.append(processed_column_data)

    processed_data_arr = np.concatenate(processed_data)

    return processed_data_arr

unpacker = BLEPacket()

# defining headers for post processing
variables = ['Acc-X', 'Acc-Y', 'Acc-Z', 'Gyro-X', 'Gyro-Y', 'Gyro-Z', 'Flex1', 'Flex2']
factors = ['mean', 'variance', 'median_absolute_deviation', 'root_mean_square', 'interquartile_range',
            'percentile_75', 'kurtosis', 'min_max', 'signal_magnitude_area', 'zero_crossing_rate',
            'spectral_centroid', 'spectral_entropy', 'spectral_energy', 'principle_frequency']

headers = [f'{var}_{factor}' for var in variables for factor in factors]

# start_time = time.time()
# print("Recording for 1 second...")
all_data = []

data = [15885, -1583, 13951, 14461, -23122, 20934, 23211, -23212]

print(data)
if len(data) == 0:
    print("Invalid data:", data)

if len(data) == 8:
    yaw, pitch, roll, accx, accy, accz, flex1, flex2 = data

    all_data.append(
        [yaw, pitch, roll, accx, accy, accz, flex1, flex2]
    )

# Convert data to DataFrame
df = pd.DataFrame([data], columns=["yaw", "pitch", "roll", "ax", "ay", "az", "flex1", "flex2"])

# Show user the data and prompt for confirmation
print(df.head())

processed_data = preprocess_data(df)

print(processed_data)
print("Processed Data Length: ", processed_data)

# Append processed data to CSV file
with open("processed_data.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerow(processed_data)

# Clear raw data list
all_data = []
print("Data processed and saved to CSV file.")
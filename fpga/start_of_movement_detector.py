import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_simulated_data():
    # simulate game movement with noise and action

    # base noise 10s long -> 20Hz*10 = 200 samples
    t = np.linspace(0, 5, 200) # Define the time range
    x1 = 0.2 * np.sin(t) + 0.2 * np.random.randn(200) 
    x1[(x1 > -1) & (x1 < 1)] = 0.0 # TODO - sensor noise within margin of error auto remove
    
    # movement motion
    period = 2  # seconds
    amplitude = 5
    t = np.linspace(0, 2, int(2 / 0.05)) # Define the time range
    x2 = amplitude * np.sin(2 * np.pi * t / period)[:40] # Compute the sine wave for only one cycle

    x = x1 
    # Add to the 40th-80th elements
    x[20:60] += x2

    x[80:120] += x2

    return x


# Define the window size and threshold factor
window_size = 11
threshold_factor = 2

# Define N units for flagging movement, 20Hz -> 2s = 40 samples
N = 40

# Initialize empty arrays for data storage
t = []
x = []
filtered = []
threshold = []
movement_detected = []
last_movement_time = -N  # set last movement time to negative N seconds ago
    
if __name__ == "__main__":
    # Create plot window
    plt.ion()
    plt.show()

    wave = generate_simulated_data()

    # Simulate incoming data every 0.05 seconds
    for i in range(len(wave)):
        # new_t = time.time()
        new_t = i
        new_x = wave[i] # TODO change to data comms line

        # process data sub-function
        # Append new data to arrays
        t.append(new_t)
        x.append(new_x)
        
        # Compute moving window median
        if len(x) < window_size:
            filtered.append(0)
        else:
            filtered.append(np.median(x[-window_size:]))
        
        # Compute threshold using past median data, threshold = mean + k * std
        if len(filtered) < window_size:
            threshold.append(0)
        else:
            past_filtered = filtered[-window_size:]
            threshold.append(np.mean(past_filtered) + (threshold_factor * np.std(past_filtered)))
        
        # Identify movement
        if len(filtered) > window_size:
            # checking if val is past threshold and if last movement was more than N seconds ago
            if filtered[-1] > threshold[-1] and t[-1] - last_movement_time >= N:
                movement_detected.append(t[-1])
                last_movement_time = t[-1]  # update last movement time
                print(f"Movement detected at {t[-1]}")
        
        # # Output data to a CSV file
        # df = pd.DataFrame({'time': t, 'original': x, 'filtered': filtered, 'threshold': threshold})
        # df.to_csv('output.csv', index=False)

        # Plot data
        plt.clf()
        plt.plot(t, x, label='original signal')
        plt.plot(t, filtered, label='filtered signal')
        plt.plot(t, threshold, label='threshold function')
        plt.legend()
        plt.draw()
        plt.pause(0.01)

        time.sleep(0.01)

    # Close plot window
    plt.close()


# t = np.linspace(0, 10, 50)
# x = np.sin(t) + 0.2 * np.random.randn(50)
# x[20] = 10
# x[10:15] = 10
# x[30:35] = 10

# # Define the window size and threshold factor
# window_size = 7
# threshold_factor = 2

# # Apply median filtering with a dynamic threshold
# filtered = np.zeros_like(x)
# threshold = np.zeros_like(x)

# # Start processing data
# for i in range(window_size, len(t)):
#     # Get the latest datapoint and add it to the filtered array
#     x_latest = np.sin(t[i]) + 0.2 * np.random.randn(1)
#     x_latest_filtered = np.median(x_latest)  # filtered latest data point

#     # Update filtered array
#     filtered = np.concatenate((filtered, [x_latest_filtered]))
#     filtered = filtered[1:]

#     # Compute threshold using past median data, threshold = mean + k * std
#     past_filtered = filtered[-window_size:]
#     threshold_value = np.mean(past_filtered) + (threshold_factor * np.std(past_filtered))
#     threshold = np.concatenate((threshold, [threshold_value]))
#     threshold = threshold[1:]

#     # Check if movement is detected
#     if filtered[-1] > threshold[-1]:
#         movement_detected = t[i]
#         print(f"Movement detected at {i}, {t[i]}")

#     # Plot the filtered signal and threshold
#     plt.clf()
#     plt.plot(t[:i+1], filtered)
#     plt.plot(t[:i+1], threshold)
#     plt.pause(0.01)

# # Compute moving window median
# for i in range(window_size // 2, len(x) - window_size // 2):
#     filtered[i] = np.median(x[i - window_size // 2:i + window_size // 2 + 1])

#     # Compute threshold using past median data, threshold = mean + k * std
#     past_filtered = filtered[i - window_size // 2:i + window_size // 2]
#     threshold[i] = np.mean(past_filtered) + (threshold_factor * np.std(past_filtered))
#     print(f"{i}, {filtered[i]}, {threshold[i]}\n")

# # Identify movement
# movement_detected = []
# for i in range(window_size // 2, len(x) - window_size // 2):
#     if filtered[i] > threshold[i]:
#         movement_detected.append(t[i])

# Output individual datapoint values to a CSV file
# df = pd.DataFrame({'time': t, 'original': x, 'threshold': threshold, 'filtered': filtered})
# df.to_csv('output.csv', index=False)

# Plot the original and filtered signals
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.plot(t, x, ':', label='Original', linewidth=2.5)
# ax.plot(t, threshold, ':', label='Threshold', color='red')
# ax.plot(t, filtered, ':', label='Filtered', linewidth=2.5)
# ax.vlines(movement_detected, ymin=filtered.min(), ymax=filtered.max(), colors='green', label='Movement Detected')
# ax.set_ylabel('Signal')
# ax.set_xlabel('Time (s)')
# ax.legend()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Set the window size and threshold factor
window_size = 25  # corresponds to 0.5s assuming sampling rate of 50 Hz
threshold_factor = 3

# Initialize the sliding window and threshold arrays
window = np.zeros(window_size)
threshold = np.zeros(window_size)

# Compute the initial threshold
median = np.median(window)
threshold[window_size // 2] = threshold_factor * np.median(np.abs(window - median))

# Initialize the flag for movement
movement_started = False

# Initialize the data array for logging
data = []

t = np.linspace(0, 10, 500)
x = np.sin(t) + 0.2 * np.random.randn(500)
x[20] = 10
x[10:15] = 10
x[30:35] = 10

for x in range(len(x)):
    # Update the sliding window and threshold arrays
    window[:-1] = window[1:]
    window[-1] = x
    median = np.median(window)
    threshold[:-1] = threshold[1:]
    threshold[-1] = threshold_factor * np.median(np.abs(window - median))
    
    # Append the current data point and window/threshold values to the data array
    data.append([x, median, threshold[window_size // 2]])
    
    # Check if the current median exceeds the threshold
    if median > threshold[window_size // 2]:
        if not movement_started:
            # Mark the start of the movement
            movement_started = True
            start_time = len(data) - 1
    else:
        if movement_started:
            # Check if the movement has lasted for at least 0.5s
            elapsed_time = len(data) - 1 - start_time
            if elapsed_time >= window_size:
                # Mark the end of the movement
                movement_started = False
                end_time = len(data) - 1
    
    # Plot the moving window and threshold lines
    plt.plot(window, label='Moving Window')
    plt.plot(threshold, 'r--', label='Threshold')
    plt.legend()
    
# Save the data array to a file
np.savetxt('data.txt', data)

# Show the final plot
plt.show()

import numpy as np
import time
import matplotlib.pyplot as plt

# Function to generate noisy sin wave
def generate_noisy_sin_wave():
    t = np.linspace(0, 10, 50)
    x = np.sin(t) + 0.2 * np.random.randn(50)
    x[20] = 10
    x[10:15] = 10
    x[30:35] = 10
    return x

if __name__ == "__main__":
    # Set the window size and threshold factor
    window_size = 7  # corresponds to 0.5s assuming sampling rate of 50 Hz
    threshold_factor = 4
    t = 0
    w = generate_noisy_sin_wave()

    # Initialize the sliding window and threshold arrays
    window = np.zeros(window_size)
    threshold = np.zeros(window_size)

    # Compute the initial threshold
    median = np.median(window)
    threshold[window_size // 2] = threshold_factor * np.median(np.abs(window - median))

    # Initialize the start time and flag for movement
    start_time = time.time()
    end_time = None
    # movement_started = False
    movement_line = np.zeros(window_size)

    # Initialize the plot
    fig, ax = plt.subplots()
    ax.set_ylim([-12, 12])
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration')
    ax.set_title('Movement Detection')

    # Plot the initial window and threshold
    x = np.arange(-window_size // 2, window_size // 2)
    line_window, = ax.plot(x, window, label='Sliding Window')
    line_threshold, = ax.plot(x, threshold, 'r--', label='Threshold')

    ax.legend()

    # Main loop for processing the live data
    with open('movement_log.txt', 'w') as f:
        while t<50:
            # Get the next data point (assuming it comes in as a scalar value)
            x = w[t]
            t += 1

            # Update the sliding window and threshold arrays
            window[:-1] = window[1:]
            window[-1] = x
            median = np.median(window)
            threshold[:-1] = threshold[1:]
            threshold[-1] = threshold_factor * np.median(np.abs(window - median))

            # Update the window and threshold lines
            line_window.set_ydata(window)
            line_threshold.set_ydata(threshold)

            # Open the log file in write mode
            
            f.write(f'Time: {time.time()}, median: {median}, thresh norm: {threshold[window_size // 2]}, thresh: {threshold[-1]}\n')
            
            # Check if the current median exceeds the threshold
            if median > threshold[window_size // 2]:
                # if not movement_started:
                    # Mark the start of the movement
                    # movement_started = True
                start_time = time.time()
                print(f'Movement started at {start_time:.2f}')
            # else:
            #     if movement_started:
            #         # Check if the movement has lasted for at least 0.5s
            #         elapsed_time = time.time() - start_time
            #         if elapsed_time >= 0.5:
            #             # Mark the end of the movement
            #             movement_started = False
            #             end_time = time.time()
            #             print(f'Movement ended at {end_time:.2f}')
            
            # Update the plot
            plt.draw()
            plt.pause(0.1)
            


# LATEST WORKING
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

t = np.linspace(0, 10, 50)
x = np.sin(t) + 0.2 * np.random.randn(50)
x[20] = 10
x[10:15] = 10
x[30:35] = 10

# Define the window size and threshold factor
window_size = 7
threshold_factor = 3

# Apply median filtering with a dynamic threshold
filtered = np.zeros_like(x)

# Compute moving window median
for i in range(window_size // 2, len(x) - window_size // 2):
    filtered[i] = np.median(x[i - window_size // 2:i + window_size // 2 + 1])

# Compute threshold using past median data
threshold = threshold_factor * np.median(np.abs(filtered - np.roll(filtered, window_size // 2)))

# Identify movement
movement_detected = []
for i in range(window_size // 2, len(x) - window_size // 2):
    if np.abs(filtered[i] - filtered[i-1]) > threshold:
        movement_detected.append(t[i])

# Output individual datapoint values to a CSV file
df = pd.DataFrame({'time': t, 'original': x, 'threshold': threshold, 'filtered': filtered})
df.to_csv('output.csv', index=False)

# Plot the original and filtered signals
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(t, x, ':', label='Original', linewidth=2.5)
ax.plot(t, threshold*np.ones_like(x), ':', label='Threshold', color='red')
ax.plot(t, filtered, ':', label='Filtered', linewidth=2.5)
ax.vlines(movement_detected, ymin=filtered.min(), ymax=filtered.max(), colors='green', label='Movement Detected')
ax.set_ylabel('Signal')
ax.set_xlabel('Time (s)')
ax.legend()
plt.show()
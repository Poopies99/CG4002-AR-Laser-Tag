import pandas as pd
import numpy as np
from collections import deque

# Initialize circular buffer and sliding window size
buffer_size = 1000 # 10s of data assuming data is coming in at 0.1s frequency
window_size = 10 # 0.5s interval

# Initialize empty deque object for circular buffer
data_buffer = deque(maxlen=buffer_size)

# Read incoming data and add to buffer
while True:
    # Read incoming data (assuming it's in format of 1x8 array)
    incoming_data = read_incoming_data()
    
    # Add incoming data to buffer
    data_buffer.append(incoming_data)
    
    # Convert buffer to pandas DataFrame for easier manipulation
    df = pd.DataFrame(data_buffer)
    
    # Preprocess the data by smoothing
    df['x_smooth'] = df.iloc[:, 0].rolling(window=window_size).mean()
    df['y_smooth'] = df.iloc[:, 1].rolling(window=window_size).mean()
    df['z_smooth'] = df.iloc[:, 2].rolling(window=window_size).mean()

    # Extract the maximum acceleration
    df['max_acc'] = np.sqrt(df['x_smooth']**2 + df['y_smooth']**2 + df['z_smooth']**2)
    
    # Set threshold as 2 sds above the running mean of the past 5 seconds of data
    past_data = df.iloc[-50:, :]
    threshold = np.mean(past_data['max_acc']) + 2 * np.std(past_data['max_acc'])
    
    # Check if latest 0.5s window is above threshold and flag as start of movement if so
    latest_data = df.iloc[-5:, :]
    latest_max_acc = np.sqrt(latest_data['x_smooth'].mean()**2 + latest_data['y_smooth'].mean()**2 + latest_data['z_smooth'].mean()**2)
    
    if latest_max_acc > threshold:
        print("Start of Movement Detected!")
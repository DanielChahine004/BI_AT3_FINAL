import time
from bitalino import BITalino
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# BITalino setup
macAddress = "0C:43:14:24:78:EA"
running_time = 100
batteryThreshold = 30
acqChannels = [0,1,2,3]
samplingRate = 100
# samplingRate = 1000
nSamples = 10

# Plotting setup
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_title("Real-time EEG Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("EEG Value")

# Data storage
max_points = running_time * samplingRate
data = deque(maxlen=max_points)
x_data = deque(maxlen=max_points)

# Connect to BITalino
device = BITalino(macAddress)

# Set battery threshold
device.battery(batteryThreshold)

# Read BITalino version
print(device.version())

# Start Acquisition
device.start(samplingRate, acqChannels)

start = time.time()
end = time.time()
sample_count = 0

def update_axes():
    if len(data) > 1:
        y_min, y_max = min(data), max(data)
        y_range = y_max - y_min
        y_buffer = y_range * 0.1  # 10% buffer
        ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    x_min, x_max = 0, len(x_data)
    x_range = x_max - x_min
    x_buffer = x_range * 0.1  # 10% buffer
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)

try:
    while (end - start) < running_time:
        # Read samples
        samples = device.read(nSamples)
        for sample in samples:
            eeg_value = sample[-1]
            data.append(eeg_value)
            x_data.append(sample_count)
            sample_count += 1
            
            # Print the EEG value for debugging
            print(f"EEG Value: {eeg_value}")
        
        # Update plot
        line.set_data(list(x_data), list(data))
        update_axes()
        plt.pause(0.001)  # Add a small pause to allow the plot to update
        
        end = time.time()

except KeyboardInterrupt:
    print("Acquisition stopped by user")

finally:
    # Stop acquisition
    device.stop()
    
    # Close connection
    device.close()
    
    # Keep the plot open
    plt.ioff()
    plt.show()
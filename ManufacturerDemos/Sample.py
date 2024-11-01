import time

from bitalino import BITalino

# This example will collect data for 5 sec.
running_time = 5

# BITalino setup
macAddress = "0C:43:14:24:78:EA"
running_time = 1000
batteryThreshold = 30
acqChannels = [0,1,2,3]
samplingRate = 100
# samplingRate = 1000
nSamples = 10

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
while (end - start) < running_time:
    # Read samples
    print(device.read(nSamples))
    end = time.time()

# Stop acquisition
device.stop()

# Close connection
device.close()
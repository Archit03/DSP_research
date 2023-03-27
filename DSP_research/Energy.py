import wave
import numpy as np
import matplotlib.pyplot as plt

# Open WAV file
with wave.open("Russia.wav", "rb") as wav_file:
    # Get parameters of WAV file
    params = wav_file.getparams()
    # Read in data from WAV file
    signal = wav_file.readframes(params[3])
    # Convert signal to array of integers
    signal = np.frombuffer(signal, dtype=np.int16)
    # Calculate signal energy
    energy = np.sum(signal**2)
    # Calculate time array
    time = np.arange(0, len(signal)) / params[2]

# Plot graph of signal energy over time
plt.plot(time, signal**2)
plt.xlabel("Time (seconds)")
plt.ylabel("Signal Energy")
print(energy)


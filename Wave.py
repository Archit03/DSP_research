import wave
import numpy as np
import matplotlib.pyplot as plt

# Open the WAV file
with wave.open('Polish Spy Numbers.wav', 'r') as wav_file:
    # Get the number of frames in the WAV file
    num_frames = wav_file.getnframes()

    # Read all the frames as a byte string
    data = wav_file.readframes(num_frames)

    # Convert the byte string to a numpy array of integers
    data = np.frombuffer(data, dtype=np.int16)

    # Get the sampling frequency of the WAV file
    sample_rate = wav_file.getframerate()

# Calculate the duration of the WAV file in seconds
duration = num_frames / float(sample_rate)

# Create a time array for the waveform plot
time = np.linspace(0, duration, num_frames)


# Plot the waveform
plt.title("Waveform analysis. ")
plt.plot(time, data)
plt.xlabel('Time (s)')
print(time)
print(data)
plt.ylabel('Amplitude')
plt.show()

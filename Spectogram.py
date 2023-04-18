import scipy.io.wavfile as wavfile
from scipy import signal
import matplotlib.pyplot as plt

# Read the WAV file
sample_rate, data = wavfile.read('9339.0kHz, USB.wav')

# Set the window size and overlap
window_size = 1024
overlap = 512

# Calculate the spectrogram
frequencies, times, spectrogram = signal.spectrogram(data, fs=sample_rate, window='hann', nperseg=window_size, noverlap=overlap)

# Plot the spectrogram
plt.pcolormesh(times, frequencies, spectrogram, cmap='jet')
plt.title("Spectrogram")
print("time,",times)
print("freq", frequencies)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

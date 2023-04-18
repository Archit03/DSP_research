import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import wave

wav = wave.open('Russia.wav', 'r')

signal = wav.readframes(-1)
signal = np.frombuffer(signal, dtype='int16')
fs = wav.getframerate()

dBFS = 20 * np.log10(np.abs(signal) / (2 ** 15))

time = np.arange(0, len(signal)) / fs
plt.plot(time, dBFS)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (dBFS)')
plt.show()


import wave
import numpy as np

# Open WAV file
with wave.open('Polish Spy Numbers.wav', 'r') as wav_file:
    # Get number of frames and frame rate
    num_frames = wav_file.getnframes()
    frame_rate = wav_file.getframerate()

    # Read audio samples as a byte string
    audio_data = wav_file.readframes(num_frames)

# Convert byte string to numpy array
audio_samples = np.frombuffer(audio_data, dtype=np.int16)

# Calculate root mean square (RMS) of audio samples
rms = np.sqrt(np.mean(np.square(audio_samples)))

# Print signal strength in decibels (dBFS)
dBFS = 20 * np.log10(rms / 32767)
print(f"Signal strength: {dBFS:.2f} dBFS")

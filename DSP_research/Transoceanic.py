import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

plt.rcParams["figure.figsize"] = (14, 6) # 14 by 6 inches graph
rate, s = wavfile.read('male.wav')
plt.plot(s)
IPython.display.Audio(s, rate=rate)

norm = 1.0 / max(np.absolute([min(s), max(s)]))
sA = 100.00 * s * norm #s SA is samples in analog audio

sD = np.round(sA) # SD is samples in the digital version

plt.plot(sA - sD) # plot the difference  

def SNR(noisy, original): #this computes signal to noise ratio(SNR)
    err = np.linalg.norm(original-noisy) #power of the error in the signal
    #power of the signal 
    sig = np.linalg.norm(original)
    #SNR in dBs
    return 10 * np.log10(sig/err)

print('SNR = %f dB' % SNR(sD , sA))
IPython.display.Audio(sA, rate=rate)

  
#TRANSMISSION ## 
'''
The function repeats a function that repeats the net effect of transmitting
audio over a cable segment transmitted by a repeated (ampliphier or regulator of a signal)

=> The signal is attenuated (reduced in power) 
=> The signal is accumulates additive noise at it propagates through the cable
=> The signal is amplified to the original amplitude by the repeater
 
''' 
 
def repeater(x, noise_amplitude, attenuation):
    noise = np.random.uniform(-noise_amplitude, noise_amplitude, len(x))
    
    #attenuation 
    x = x * attenuation 
    
    #noise
    x = x + noise
    # gain compensation
    
    return x / attenuation

def analog(x, num_repeaters, noise_amplitude, attenuation):
    
    for n in range(0, num_repeaters):
        x = repeater(x, noise_amplitude, attenuation)
    return x

def digital(x, num_repeaters, noise_amplitude, attenuation):
    for n in range(0, num_repeaters):
        x = np.round(repeater(x, noise_amplitude, attenuation))
    return x

num_repeaters = 70
noise_amplitude = 0.2
attenuation = 0.5

yA = analog(sA, num_repeaters, noise_amplitude, attenuation)
print("Analog transmission: SNR = %f dB" % SNR(yA, sA))

yD = digital(sD, num_repeaters, noise_amplitude, attenuation)
print("Digital transmission: SNR = %f dB" % SNR(yD, sA))

IPython.display.Audio(yA, rate=rate)
IPython.display.Audio(yD, rate=rate)

noise_amplitude = 0.3 

yA = analog(sA, num_repeaters, noise_amplitude, attenuation)
print('Analog transmission: SNR = %f dB ' % SNR(yA, sA))

yD = digital(sD, num_repeaters, noise_amplitude, attenuation)
print("Digital transmission: SNR=  %f dB " % SNR(yD, sD))







import numpy as np 
import IPython
from numpy.lib.financial import rate
from scipy.io import wavfile
import numba 
from numba import prange


def Euclid(A ,B):
    a = A 
    b = B 
    while a!= b:
        if a > b:
            a = a - b 
        else:
            b = b - a
    return A // a, B // b 

def setup_filters(output_rate, input_rate):
    No, Ni = Euclid(output_rate, input_rate) #No/Ni is input/output samples respectively
    filter_bank = [[] for i in range(Ni)]
    #while output index spans [0, No-1], the input spans[0,Ni-1]
    for m in prange(0,No):
        anchor = int(m * (Ni / No) + 0.5)
        tau = (m * Ni / No) - anchor
        filter_bank[anchor].append([tau *(tau - 1)/ 2, (1 - tau) * (1 + tau), tau * (tau + 1) / 2])
    return filter_bank 

setup_filters(4, 5)
                                   
def resample(output_rate, input_rate, x):
    No, Ni = Euclid(output_rate, input_rate)
    filter_bank = setup_filters(No, Ni)
    
    y = np.zeros(No * len(x)) // Ni 
    m = 0 
    for n, x_n in enumerate(x[1:-1]):
        for fb in filter_bank[n % Ni]:
            y[m] = x[n-1] * fb[0] + x[n] * fb[1] + x[n + 1] * fb[2]
    return y 

x = np.cos(2 * np.pi * 440 / 44100 * np.arange(0, 20000))
IPython.display.Audio(x, rate =44100)

y = resample(12000 , 44100, x)
IPython.display.Audio(y, rate=12000)


import matplotlib.pyplot as plt 
import numpy as np 
import IPython 
import scipy.signal as sp 
from scipy.io import wavfile

plt.rcParams['figure.figsize'] = (14, 4)
SF , s  = wavfile.read("download.wav")
IPython.display.Audio(s , rate=SF) 

fc = 200.00
wc = fc / (SF/2)
b , a  = sp.butter(6 , wc )
'''
A butterworth filter is designed to to have a freq response as flat 
as possible in the passband   
'''
wb, Hb = sp.freqz(b, a , 1024)
plt.plot(wb[0:200]/np.pi *(SF/2) , np.abs(Hb[0:200]))

y = sp.lfilter(b , a , s)
IPython.display.Audio(data=y , rate=SF)

tb = 100
# length of the filter
M = 1200
h = sp.remez(M, [0, fc, fc+tb, SF/2], [1, 0], [1, 1], Hz=SF, maxiter=50)

w, H = sp.freqz(h, 1, 1024)
plt.semilogy(w[0:200]/np.pi * (SF/2), np.abs(H[0:200]))
plt.plot(wb[0:200]/np.pi * (SF/2), np.abs(Hb[0:200]), 'green')
IPython.display.Audio(y , rate=SF)

fh = 4000 
M = 1601 
hh = sp.remez(M , [0, fh- tb, fh, SF/2] , [0,1], [10, 1], Hz=SF, maxiter=50)

w, HH = sp.freqz(hh , 1, 1024)
plt.semilogy(w/np.pi * (SF/2), np.abs(HH))

y = sp.lfilter(hh , [1], s)
IPython.display.Audio(data=y , rate=SF)

from functools import singledispatchmethod
from IPython.terminal.embed import embed
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import IPython
from scipy.io import wavfile
from scipy import signal

plt.rcParams['figure.figsize'] = (14, 4)
fs, data = wavfile.read('download.wav')

data = data / 32767.0
IPython.display.Audio(data=data, rate=fs, embed=True)

'''
x[n] --> + --> G -----> A-----> y[n]
     /|\                     /|\ 
      |----------F<-----------|
      
The guitar G, including the possibility of driving the string vibration during oscillation
      
'''

plt.plot(data)
plt.xlabel("sample")
plt.ylabel("amplitude")

s = abs(np.fft.fft(data[10000:40000]))
s = s[0:int(len(s)/2)]
plt.plot(np.linspace(0, 1, len(s)) * (fs/2), s)
plt.xlabel("freq(Hz")
plt.ylabel("magnitude")

w, h = signal.freqz(1, [1, 0, 0, 0, 0, 0, 0, 0, 0, -.99**9])
plt.plot(w , abs(h))

# y[n] = p^N y[n-N] + x[n] - px[n-1]
class guitar:
    def __init__(self, pitch=110, fs=24000):
        # init the class with desired pitch and underlying sampling frequency
        self.M = int(np.round(fs / pitch)) # fundamental period in samples
        self.R = 0.9999   # decay factor
        self.RM = self.R ** self.M  
        self.ybuf = np.zeros(self.M)  # output buffer (circular)
        self.iy = 0                   # index into out buf
        self.xbuf = 0                 # input buffer (just one sample)
        
    def play(self, x):
        y = np.zeros(len(x))
        for n in range(len(x)):
            t = x[n] - self.R * self.xbuf + self.RM * self.ybuf[self.iy]
            self.ybuf[self.iy] = t
            self.iy = (self.iy + 1) % self.M
            self.xbuf = x[n]
            y[n] = t
        return y     
    
    # create a 2-second signal
d = np.zeros(fs*2)
# impulse in zero (string plucked)
d[0] = 1

# create the A string
y = guitar(110, fs).play(d)
IPython.display.Audio(data=y, rate=fs, embed=True)

s = abs(np.fft.fft(y))
s = s[0:int(len(s)/2)]
plt.plot(np.linspace(0,1,len(s))*(fs/2), s)

#To get rid of the unwanted spectral content. 
#using a lowpass filter, use scipy's butterworth filter.

class guitar:
    def __init__(self, pitch=110 , fs=24000):
        self.M = int(np.round(fs/pitch))
        self.R = 0.9999 # decay factor 
        self.RM = self.R ** self.M  
        self.ybuf = np.zeros(self.M) #output buffer(circular)
        self.iy = 0 #index into out buf  
        self.xbuf = 0 #input buffer(just one sample)
        #6th order butterworth, keep 5 harmonics:  
        self.bfb , self.bfa = signal.butter(6 , min(0.5 , 0.5 * pitch / fs))
        self.bfb *= 1000 
        self.bfs = signal.lfiltic(self.bfb, self.bfa , [0]) 
        
    def play(self, x):
        y = np.zeros(len(x))
        for n in range(len(x)):
            #comb filter 
            t = x[n] - self.R * self.xbuf + self.RM * self.ybuf[self.iy]
            self.ybuf[self.iy] = t
            self.iy = (self.iy + 1) % self.M
            self.xbuf = x[n]
            y[n] , self.bfs = signal.lfilter(self.bfb , self.bfa , [t] , zi=self.bfs)
        return y     
    
d = np.zeros(fs*2)
d[0] = 1 

y = guitar(110 , fs).play(d)    
IPython.display.Audio(data=y , rate=fs, embed=True)

s = abs(np.fft.fft(y[10000:30000])) 
s = s[0:int(len(s)/2)]
plt.plot(np.linspace(0 , 1, len(s)) * (fs/2) , s)

def amplify(x):
    TH = 0.9 #threshold 
    y = np.copy(x)
    y[ y > TH] = TH 
    y[y < -TH] = -TH 
    return y 

x = np.linspace(-2 , 2 , 100)
plt.plot(x , amplify(x))
plt.xlabel("input")
plt.ylabel("output")

'''   
simulation of an acoustic guitar,
The output of the acoustic channel for a guitar amplifier 
distance of d meters will be there . Hence y[n] = ax[n-M], 
where (decay factor)a = 1/d and M is the propagation delay in Samples  
M = ceil(d/cF_s) , where c is speed of sound and F_s is freq. 
'''

class feedback:
    SPEED_OF_SOUND = 343.0 #m/s
    def __init__(self, max_distance_m = 5, fs=24000):
        self.L = int(np.ceil(max_distance_m / self.SPEED_OF_SOUND * fs))
        self.xbuf = np.zeros(self.L)
        self.ix = 0 
        
    def get(self , x , distance):
        d = int(np.ceil(distance / self.SPEED_OF_SOUND * fs))
        self.xbuf[self.ix] = x 
        x = self.xbuf[(self.L + self.ix - d) % self.L]   
        self.ix = (self.ix + 1 ) % self.L  
        return x / float(distance)
    
g = guitar(110)
f = feedback()

coupling_loss = 0.0001  

START_DISTANCE = 3  
END_DISTANCE   = 0.05 

N = int(fs * 5)
y = np.zeros(N)
x = [1]

for n in range(N):
    y[n] = amplify(g.play(x))
    x = [coupling_loss * f.get(y[n] , START_DISTANCE if n < (1.5 * fs) else END_DISTANCE)]
    
    
IPython.display.Audio(data=y , rate=fs , embed=True)

import matplotlib.pyplot as plt 
import numpy as np 
import scipy.signal as sp 
import IPython
from scipy.io import wavfile 

plt.rcParams['figure.figsize'] = (14, 4)


'''
x[n]----->+ ----------------------------->y[n]
          |                           |
          |(alpha)                    | 
          |<-----------H(z)----z^-M---| 
          
          where alpha is attenuation where signal of an echo dies out over time
          
          y[n] = x[n]- lamda * x[n-1] + lamda * y[n-1] + (alpha)(1-lamda) * y[n-M]
'''

#using the leaky integrator for echo cancelation 
def echo(x, M, lmb=0.6, alpha=-0.8):
    # if the first argument is a scalar, assume the input is a delta sequence of length x
    #  in this case, the function returns the truncated impulse response of the room.
    if np.isscalar(x):
        x = np.zeros(int(x))
        x[0] = 1
    y = np.zeros(len(x))
    for n in range(0, len(x)):
        if n >= M:
            y[n] = x[n] - lmb * x[n-1] + lmb * y[n-1] + alpha * (1 - lmb) * y[n - M]
        elif n > 0:
            y[n] = x[n] - lmb * x[n-1] + lmb * y[n-1]
        else:
            y[n] = x[n]
    return y

plt.plot(echo(1000, 100))


Fs , s = wavfile.read('male.wav')
s = s / 32767.0 # scale the signal to floats in [-1, 1]
print('Sampling rate: ', Fs,  'Hz, data length: ',  len(s), 'samples')
IPython.display.Audio(s, rate=Fs)            

es = echo(s, int(0.020 * Fs))
IPython.display.Audio(es, rate=Fs)

IPython.display.Audio(np.r_[es, es - s], rate=Fs)

def lms(x, d, N, a=0.001):
    # Run the LMS adaptation using x as the input signal, d as the desired output signal and a as the step size
    # Will return an N-tap FIR filter
    #
    # initial guess for the filter is a delta
    h = np.zeros(N)
    h[0] = 1
    # number of iterations
    L = min(len(x), len(d))
    # let's store the error at each iteration to plot the MSE
    e = np.zeros(L)
    # run the adaptation
    for n in range(N, L):
        e[n] = d[n] - np.dot(h, x[n:n-N:-1])
        h = h + a * e[n] * x[n:n-N:-1]
    return h, e[N:]

# echo delay
delay = 100

# LMS parameters
taps = 500
step_size = 0.0008

# this function generates runs the LMS adaptation on a signal of length L and returns the filter's coefficients
def test_lms(L):
    # the input signal
    ns = np.random.randn(L)
    return lms(ns, echo(ns, delay), taps, step_size)[0]
    
h = echo(taps, delay)
# precision increases with length of the adaptation 
plt.plot(h, 'g'); # original impulse response (green)
plt.plot(test_lms(1000), 'r');
plt.plot(test_lms(5000), 'y');
best = test_lms(10000)
plt.plot(best, 'b');

plt.plot(h - best)

Trails = 100


TRIALS = 100
L = 8000

for n in range(0, TRIALS):
    ns = np.random.randn(L)
    err = np.square(lms(ns, echo(ns, delay), taps, step_size)[1]) 
    if n == 0:
        mse = err
    else:
        mse = mse + err
mse = mse / TRIALS   
plt.plot(mse);
            
delay = int(0.020 * Fs)
audio = np.r_[s, s, s, s, s] 

taps = 1500 
step_size = 0.021 
h , err = lms(audio, echo(audio, delay), taps, step_size)

plt.plot(echo(len(h), delay))
plt.plot(h)
           
plt.plot(echo(len(h), delay) - h)

es = echo(s, delay)
IPython.display.Audio(np.r_[es, es - s, es - sp.lfilter(h, 1, s)], rate=Fs)

plt.plot(np.r_[es, es - s , es - sp.lfilter(h, 1, s)])
           

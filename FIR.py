import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# implementation of a causal filter for finite impulse response


class FIR_loop():
    def __init__(self, h):
        self.h = h
        self.ix = 0  # index into the samples
        self.M = len(h)
        self.buf = np.zeros(self.M)

    def filter(self, x):
        y = 0
        self.buf[self.ix] = x
        for n in range(0, self.M):
            y += self.h[n] * self.buf[(self.ix+self.M-n) % self.M]
        self.ix = (self.ix + 1) % self.M
        return y


h = np.ones(5)/5

f = FIR_loop(h)
for n in range(0, 10):
    print(f.filter(n), end=", ")

M = 5
h = np.ones(M)/float(M)

x = np.concatenate((np.arange(1, 9), np.ones(5) * 8, np.arange(8, 0, -1)))
plt.stem(x)
print("\n Signal length: ", len(x))
y = np.convolve(x, h, mode='valid')
print('signal length: ', len(y))
plt.stem(y)

y = np.convolve(x, h, mode='same')
print('signal length:', len(y))
plt.stem(y)


'''
The another way of embedding a finite-length signal to build is a periodic extension. 
y[n] = (M-1)sum(k=0) h[k]x[n-k]
'''


def cconv(x, h):
    L = len(x)
    xp = np.concatenate((x, x))
    y = np.convolve(xp, h)
    return y[L:2*L]


y = cconv(x, h)
print('signal length:', len(y))
plt.stem(y)

y = cconv(np.concatenate((x, np.zeros(M-1))), h)
print('signal length:', len(y))
plt.stem(y)
plt.stem(y - np.convolve(x, h, mode='full'), markerfmt='ro')


'''
Anti-causal implementation of DFT in the time domain.
(x (convolute) y)[n] = IDTFT(X(e^jw)Y(e^jw)[n])

'''
x = np.concatenate((np.arange(1, 9), np.ones(5) * 8, np.arange(8, 0, -1)))
import numpy as np
def dftconv(x , h, mode='full'):
    N = len(x)
    M = len(h)
    X = np.fft.fft(x , n=N+M-1)
    H = np.fft.fft(h , n=N+M-1)
    
    y = np.real(np.fft.ifft(X * M))
    if mode == 'valid':
        return y[M-1:N]
    
    elif mode == 'same':
        return y [int((M-1)/2):int((M-1)/2)+N]
    else: 
        return y
    
y = np.convolve(x , h , mode='valid')
print('Signal length:', len(y))
plt.stem(y)


y = dftconv(x, h, mode='valid')
print('signal length: ' , len(y))     
plt.stem(y,markerfmt='ro')


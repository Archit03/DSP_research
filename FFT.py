import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (14, 4)
'''
using the direct and inverse DFT
'''
def dft_matrix(N):
    a = np.expand_dims(np.arange(N), 0)
    
    w  = np.exp(-2j *(np.pi / N) * (a.T * a))
    
    return w

x = np.array([5, 7, 9])

N = len(x)
w = dft_matrix(N)

X = np.dot(w, x)
x_hat = np.dot(w.T.conjugate(), X) / N #IDFT 

print(x_hat)

N = 128
x =np.zeros(N)
x[0:64] = 1 
plt.stem(x)

w = dft_matrix(N)

X = np.dot(w,x)

plt.stem(abs(X))
plt.show()

plt.stem(np.angle(X))

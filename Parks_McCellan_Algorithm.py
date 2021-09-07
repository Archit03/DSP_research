import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio , display
from ipywidgets import interactive, fixed
import numba 

plt.rcParams["figure.figsize"] = (14 , 4)
 
@numba.jit(nopython=True)
def MSE_fit(A , B , order):
 #order => number of taps(samples) 
 #A => passband 
 #B => stopband
    if order < 3:
        raise  ValueError("Order can't be less than 3")
    # interpolation points always one more than the order of the interpolator
    pts = order + 1 
    
    # split number of the interpolation  points across intervals proportionately 
    #with the length of each interval 
    
    ptsA = int(pts * A / (A+(1-B)))
    if ptsA < 2:
        ptsA = 2 
    ptsB = pts - ptsA
    
    #for the mse fit, place a point at each interval edge and distribute the rest 
    #(if any) evenly over the interval 
    
    x = np.concatenate((np.arange(0 , ptsA) * (A / (ptsA-1)) , B + np.arange(0 , ptsB) * ((1 -B)/(ptsB - 1))))
    y = np.concatenate((np.ones(ptsA)) , (np.zeros(ptsB)))
    
    p = np.poly1d  (np.polyfit(x , y , order))
    return p , x , y 
 
def MSE_fit(A=0.4 , B=0.6 , order=10):
    p, x, y = MSE_fit(A , B , order)
    
    t = np.linspace(0 , 1 , 100) 
    lims = [(0,1 , -.5,1.5) , (0,A , 0.8,1.2), (B,1,-0.2,0.2)]
    for n, r in enumerate(lims):
        plt.subplot(1 , 3, n+1)
        plt.plot((0, A), (1,1), 'red', (B,1), (0,0) , 'red' , x,y , 'oy' , t , p(t), '_')
        plt.xlim(r[0], r[1])
        plt.ylim(r[2], r[3])
        
v = interactive(MSE_fit , order=(3,30) , A=(0.0 , 0.5), B=(0.5, 1.0))

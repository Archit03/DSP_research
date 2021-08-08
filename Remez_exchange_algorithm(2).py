import numpy as np 
import matplotlib.pyplot as plt 
from IPython.display import Audio, display 
from ipywidgets import interactive , fixed 
plt.rcParams['figure.figsize'] = (14, 4)


def find_error(p, A, B):
    interval = {(0,A):1 , (B,1):0 }
    loc = [] #location of the extrema  
    err = [] #values of the extrema 
    for rng, val in interval.items():
        t = np.linspace(rng[0], rng[1], 100)
        y = val - p(t)
        
        ix = np.diff(np.sign(np.diff(y))).nonzero()[0] + 1 
        loc = np.hstack((loc, t[0], t[ix], t[-1]))
        err = np.hstack((err , y[0] , y[ix] , y[-1]))
    return loc, err 

def remez_fit(A, B, Aweight, order, iterations):
    def weigh(x):
        #the weighting function 
        if np.isscalar(x):
            return 1.0/Aweight if x <= A else 1 
        return [1.0/Aweight if v <= A else 1 for v in x] 
    
    pts = order + 2 
    ptsA = int(pts * A / (A-B+1))
    if ptsA < 2:
        ptsA = 2 
    ptsB = pts - ptsA     
    x = np.concatenate((np.arange(1, ptsA+1) * (A/ (ptsA+1)), B + np.arange(1 ,  ptsB+1) * ((1-B)/ (ptsB + 1))))
    y = np.concatenate((np.ones(ptsA), np.zeros(ptsB)))
     
    data = {} 
    for n in range(0, iterations):
        data['prev_x'] = x 
        data['prev_y'] = y  
        
        V = np.vander(x, increasing=True) 
        V[:, -1] = pow(-1, np.arange(0, len(x))) * weigh(x)
        v = np.linalg.solve(V , y)
        p = np.poly1d(v[-2::-1])
        e = np.abs(v[-1]) 
        data['Aerr'] = e / Aweight 
        data['Berr'] = e 
        
        loc, err = find_error(p, A, B)
        alt = [] 
        for n in range(0, len(loc)):
            c = {
                'loc': loc[n], #locations of the extrema 
                'sign' : np.sign(err[n]), # sigh(error(n)) ,  values of the extrema 
                'err_mag': np.abs(err[n]) / weigh(loc[n]) #error magnitude 
            }
            if c['err_mag'] >= e - 1e-3:
                if alt == [] or alt[-1]['sign'] != c['sign']:
                    alt.append(c)
                elif alt[-1]['err_mag'] < c['err_mag']:
                    alt.pop() 
                    alt.append(c)
        while len(alt) > order + 2 :
            if alt[0]['err_mag'] > alt[-1]['err_mag']:
                alt.pop(-1)
            else:
                alt.pop(0)
        x = [c['loc'] for c in alt]
        y = [1 if c <= A else 0 for c in x]
        data['new_x'] = x 
        
    return p, data 
                

def remez_fit_show(A=0.4, B=0.6, Aweight=50, order=10, iterations=1):
    p, data = remez_fit(A, B, Aweight, order, iterations)
    
    t = np.linspace(0, 1, 300) 
    Ae = data['Aerr'] #Error in A 
    Be = data['Berr'] #Error in B 
    
    def loc_plot(A, B, data):
        plt.plot((0,A), (1,1), 'red', (B,1), (0,0), 'red', (0,A), (1+Ae, 1-Ae), 'cyan', (0,A), (1-Ae, 1-Ae), 'cyan', (B,1), (Be,Be), 'cyan', (B,1), (-Be, -Be), 'cyan', data['prev_x'], data['prev_y'], 'oy', data['new_x'], p(data['new_x']), '*', t, p(t), '_')
    loc_plot(A, B, data)
    plt.show()
    
    lims = [(0,A, 1-1.5*Ae, 1+1*Ae), (B, 1, -1.5*Be, 1.5*Be)]
    for n, r in enumerate(lims):
        plt.subplot(1,2, n+1)
        loc_plot(A, B, data)
        plt.xlim(r[0], r[1])
        plt.ylim(r[2], r[3])
        
v = interactive(remez_fit_show, A=(0.0, 0.5, 0.4), B=(0.5, 1.0), Aweight=(1,100,10), order=(5,20), iterations=(1,10))
display(v)
            
                    
                
                                 
                        
     
     
        
            

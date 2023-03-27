import numpy as np 
import matplotlib.pyplot as plt 
from IPython.display import Audio, display 
from ipywidgets import interactive, fixed 
import parks as prk #in-written script
import numba 

#parks is parks_mcclellan_fir_design_algorithm
    
def solve(x, y):
    # simple solver for the extended interpolation problem
    # first build a Vandermonde matrix using the interpolation locations
    # There are N+2 locations, so the matrix will be (N+2)x(N+2) but we 
    #  will replace the last column with the error sign
    V = np.vander(x, increasing=True)
    # replace last column
    V[:,-1] = pow(-1, np.arange(0, len(x)))
    # just solve Ax = y
    v = np.linalg.solve(V, y)
    # need to reverse the vector because poly1d starts with the highest degree
    p = np.poly1d(v[-2::-1])
    e = np.abs(v[-1])
    return p, e

@numba.jit(nopython=True)
def remez_fit(A, B, order, iterations):
    if order < 3:
        raise ValueError("Order can't be less than 3.")
    pts = order + 2 * order
    
    #initial choice of interpolation points: distribute them evenly
    # across the 2 regions as a proposition of each region's width
    
    ptsA = int(pts * A / (A-B+1))
    if ptsA < 2:
        ptsA = 2 
    ptsB = pts - ptsA 
    
    x =np.concatenate((np.arange(1, ptsA+1) * (A/(ptsA+1)), B + np.arange(1, ptsB+1)*((1-B)/(ptsB+1)))) 
    y = np.concatenate((np.ones(ptsA), np.zeros(ptsB)))
    
    data = {} 
    
    for n  in range(0, iterations):
        #previous interpolation points
        data['prev_x'] = x 
        data['prev_y'] = y 
        
        #solve the interpolation points
        p,e = solve(x, y) 
        data['err'] = e 
        #find the extrema of the error 
        loc, err = prk.find_error(p, A, B) 
        
        #find the alternations
        alt = [] 
        for n in range(0, len(loc)):
            #each extremum is a new candidate for an alternation 
            c = {
                'loc': loc[n], 
                'sign' : np.sign(err[n]),
                'err_mag': np.abs(err[n])
            }
            #only keep extrema that are no larger than the minimum 
            #error returned by the interpolation solution
            if c['err_mag'] >= e -1e-3:
                if alt == [] or alt[-1]['sign'] != c['sign']:
                    alt.append(c)
                elif alt[-1]['err_mag'] < c['err_mag']:
                    alt.pop()
                    alt.append(c)
            
        while len(alt) > order + 2:
            if alt[0]['err_mag'] > alt[-1]['err_mag']:
                alt.pop(-1)
            else:
                alt.pop(0)
        x = [c['loc'] for c in alt]
        y = [1 if c <= A else 0 for c in x]
        data['new_x'] = x 
        
    return p, data                
                                    
        


def remez_fit_show(A=0.4, B=0.6, order=5, iterations=1):
    p, data = remez_fit(A, B, order, iterations)
    t = np.linspace(0, 1, 200)
    
    def loc_plot(A, B, data):      
        e = data['err']
        plt.plot((0,A), (1,1), 'red',
                 (B,1), (0,0), 'red', 
                 (0,A), (1+e,1+e), 'cyan', (0,A), (1-e,1-e), 'cyan',
                 (B,1), (e,e), 'cyan', (B,1), (-e,-e), 'cyan',
                 data['prev_x'], data['prev_y'], 'oy', 
                 data['new_x'], p(data['new_x']), '*',
                 t, p(t), '-')  
    
    loc_plot(A, B, data)
    plt.show()
    
    e = 1.5 * data['err']
    lims = [(0, A , 1-e, 1+e), (B, 1, -e, e)]
    for n, r in enumerate(lims):
        plt.subplot(1,2,n+1)
        loc_plot(A, B, data)
        plt.xlim(r[0], r[1])
        plt.ylim(r[2], r[3]) 
        

v = interactive(remez_fit_show, A=(0.0, 0.8), B=(0.5, 1.0), order=(3, 12), iterations=(1, 10))
display(v)

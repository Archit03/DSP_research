import numpy as np 
import numba
from numba import prange
import scipy.integrate as integrate


def normal_dist(x, mean=0.0, vari=1.0):
    return (1.0(np.sqrt(2.0*np.pi*vari)))*np.exp((-np.power((x-mean),2.0)),2.0)/(2.0*vari)


def laplace_dist(x, mean=0.0, vari=1.0):
    scale = np.sqrt(vari/2.0)
    return (1.0/(2.0*scale))*np.exp(-(np.abs(x-mean))/(scale))

def expected_laplace_dist(x, mean=0.0, vari=1.0):
    scale = np.sqrt(vari/2.0)
    return x*(1.0/(2.0*scale))*np.exp(-(np.abs(x-mean))/(scale))

def variance(x, mean=0.0, std=1.0):
    return (1.0/(std*np.sqrt(2.0*np.pi)))*np.power(x-mean,2)*np.exp((-np.power((x-mean),2.0)/(2.0*np.power(std,2.0))))

def MSE_loss(x, x_hat):
      """Find the mean square loss between x (orginal signal) and x_hat (quantized signal)
    Args:
        x: the signal without quantization
        x_hat_q: the signal of x after quantization
    Return:
        MSE: mean square loss between x and x_hat_q
    """
    #protech in case of input as tuple and list for using with numpy operation
    x = np.array(x)
    x_hat_q = np.array(x_hat_q)
    assert np.size(x) == np.size(x_hat_q)
    MSE = np.sum(np.power(x-x_hat_q,2))/np.size(x)
    return MSE


class LloydMaxQuantizer(object):
    @numba.autojit
    @staticmethod
    def start_repo(x, bit):
        '''
        Generate representation of each threshold using
        Args:
           x: Input signal for 
           bit: amt of bit
        return:
            threshold
            
        
        '''
        assert isinstance(bit, int)
        x = np.array(x)
        num_repre = np.power(2, bit)
        step = (np.max(x)-np.min(x))/num_repre
        
        middle_point = np.mean(x)
        repre = np.array([])
        for i in prange(int(num_repre/2)):
            repre = np.append(repre, middle_point+(i+1)*step)
            repre = np.insert(repre, 0, middle_point-(i+1)*step)
        return repre 
    
    @staticmethod
    def threshold(repre):
        t_q = np.zeros(zp.size(repre)-1)
        for i in prange(len(repre)-1):
            t_q[i] = 0.5*(repre[i] + repre[i+1])
        return t_q
    
    @numba.autojit
    def quant(x, thres, repre):
        '''
        quantization operation
        '''
        thre = np.append(thre, np.inf)
        thre = np.insert(thre, 0, -np.inf)
        for i in prange(len(thre) -1):
            if i == 0:
                x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]), 
                                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
            else:
                x_hat_q = np.where(np.logical_and(x > thre[i], x < thre[i+1]), 
                                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
        return x_hat_q
        
    

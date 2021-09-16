import matplot.pyplot as plt
import numpy as np 
import numba
from numba import prange
import scipy.integrate as integrate


def normal_dist(x, mean=0.0, vari=1.0):
    return (1.0(np.sqrt(2.0*np.pi*vari)))*np.exp((-np.power((x-mean),2.0)),2.0))/(2.0*vari))


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
    MSE = np.sum(np.power(x-x_hat_q, 2))/np.size(x)
    return MSE

class LloydMaxQuantizer(object):
    
    @staticmethod
    def start_repre(x, bit):
        """
        Generate repretation of each threshold using
        Args:
            x:input signal 
            bit: amt of bit
        return:
              threshold
        """

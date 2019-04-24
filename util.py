import numpy as np


def sigmoid(val):
    return(1./(1.+np.exp(-val)))


def ReLU(arr):
    return [max(0, val) for val in arr]


def copy_w_err(bitarr, prob=0.0001):
    out = bitarr
    for i in range(len(out)):
        if np.random.binomial(1, prob) == 1:
            val = np.random.uniform()
            # if val < 0.33333333333:
            #     out[i] = -1
            # elif val < 0.6666666667:
            if val < 0.5:
                out[i] = 0
            else:
                out[i] = 1
    return out

import numpy as np
from util import *


class PopulationModel(object):
    """ Population Model for testing TW vs. LRC """
    def __init__(self, ms, fs, N, v):
        super(PopulationModel, self).__init__()
        self.ms = ms
        self.fs = fs
        self.N = N
        self.v = v

    def set_f_func(self, f_func):
        self.f_func = f_func

    def set_g_func(self, g_func):
        self.g_func = g_func

    def inherit_w(self, w_m, w_f):
        if type(w_f) == float:
            if np.random.binomial(1, 0.5) == 1:
                return np.random.normal(w_m, 0.1)
            else:
                return np.random.normal(w_f, 0.1)
        else:
            w_k = []

            for i in range(len(w_m)):
                if np.random.binomial(1, 0.5) == 1:
                    w_k.append(w_m[i])
                else:
                    w_k.append(w_f[i])

            return copy_w_err(w_k)

    def inherit_theta(self, theta_m, theta_f):
        theta_k = []

        for i in range(len(theta_m)):
            if np.random.binomial(1, 0.5) == 1:
                theta_k.append(theta_m[i])
            else:
                theta_k.append(theta_f[i])

        return copy_w_err(theta_k)

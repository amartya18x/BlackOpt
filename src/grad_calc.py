import numpy as np


class grad_calculator(object):

    def __init__(self, func, num_var):
        self.func = func
        self.num_var = num_var
        self.h = 10e-2

    def get_gradient(self, curr_var):
        grads = np.zeros(self.num_var)
        for i in range(0, self.num_var):
            delta = np.zeros([self.num_var]).astype(np.float32)
            delta[i] = self.h
            f_xph = self.func(curr_var + delta)
            f_xmh = self.func(curr_var - delta)
            grads[i] = f_xph - f_xmh
        return grads / self.h

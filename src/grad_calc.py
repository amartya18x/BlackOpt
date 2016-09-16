import numpy as np


class grad_calculator(object):

    def __init__(self, func, num_var):
        self.func = func
        self.num_var = num_var
        self.h = 10e-3

    def get_gradient(self, curr_var):
        grads = np.zeros(self.num_var)
        for i in range(0, self.num_var):
            delta = curr_var
            delta = np.copy(curr_var).astype(np.float32)
            delta[i] = delta[i] + self.h
            f_xh = self.func(delta)
            f_x = self.func(curr_var)
            grads[i] = f_xh - f_x
        return grads / self.h

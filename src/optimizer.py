from grad_calc import grad_calculator
import numpy as np


class optim(object):

    def __init__(self, func, eps, num_v):
        self.func = func
        self.eps = eps
        self.lr = 10e-5
        self.num_var = num_v
        self.gc = grad_calculator(self.func,num_v )

    def grad_desc(self, x):
        return self.gc.get_gradient(x)

    def initialize(self):
        return np.random.random([self.num_var])

    def optimize(self):
        x = self.initialize()
        grad_x = self.grad_desc(x)
        cost = [self.func(x)]
        t = 0
        while(np.sum(grad_x)**2 > self.eps and t < 100000):
            x_new = x - self.lr * grad_x
            cost.append(self.func(x_new))
            grad_x = self.grad_desc(x_new)
            x = x_new
            t +=1
        return cost

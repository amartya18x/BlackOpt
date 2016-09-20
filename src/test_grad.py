from grad_calc import grad_calculator
import numpy as np
from optimizer import optim
import matplotlib.pyplot as plt

test_var = 10


def x_sq(x):
    return x[0] * x[0]


def x_sq_grad(x):
    return 2 * x[0]

tup_x_sq = (x_sq, x_sq_grad)


def x_sq_y(var):
    return var[0] ** 2 + 2 * (var[1] - 1)**2 + 1


def comp_func(var):
    y = var[1]
    x = var[0]
    return (x - y)**2 - np.log((x + y)**2)


def x_sq_y_grad(var):
    return [2 * var[0], 4 * (var[1] - 1)]


tup_x_sq_y = (x_sq_y, x_sq_y_grad)

grad_obj1 = grad_calculator(tup_x_sq[0], 1)
grad_obj2 = grad_calculator(tup_x_sq_y[0], 2)

error = 0
for i in range(0, test_var):
    test_variable = [(np.random.random() - 0.5) * 100.0]
    evaluated_grad = grad_obj1.get_gradient(test_variable)
    actual_grad = tup_x_sq[1](test_variable)
    error = error + np.sum(evaluated_grad - actual_grad) ** 2

print(error / test_var)

error = 0
for i in range(0, test_var):
    test_variable1 = (np.random.random() - 0.5) * 100.0
    test_variable2 = (np.random.random() - 0.5) * 100.0
    test_variable = [test_variable1, test_variable2]
    evaluated_grad = grad_obj2.get_gradient(test_variable)
    actual_grad = tup_x_sq_y[1](test_variable)
    error = error + np.sum(evaluated_grad - actual_grad) ** 2

print(error / test_var)

optim = optim(tup_x_sq_y[0], 10e-10, 2)
cost = optim.optimize()
plt.plot(cost)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random


def lin_func(x: list, a: float, b: float) -> np.array:
    """
    Function for finding result of linear function for every x

    :param x: list of elements
    :param a: coefficient a
    :param b: coefficient b
    :type x: list
    :type a: float
    :type b: float
    :return: array of results of counting
    :rtype: np.array
    """

    return a * np.array(x) + b


def ration_func(x: list, a: float, b: float) -> np.array:
    """
    Function for finding result of rational function for every x

    :param x: list of elements
    :param a: coefficient a
    :param b: coefficient b
    :type x: list
    :type a: float
    :type b: float
    :return: array of results of counting
    :rtype: np.array
    """

    return a / (1 + b * np.array(x))


def errors_func_lin(x: list, y: list, a: float, b: float) -> np.array:
    """

    """

    return np.sum(((a * np.array(x) + b) - np.array(y)) ** 2)


def derivative_a_errors_func_lin(x: list, y: list, a: float, b: float) -> np.array:
    return 2 * np.sum((a * np.array(x) + b - np.array(y)) * np.array(x))


def derivative_b_errors_func_lin(x: list, y: list, a: float, b: float) -> np.array:
    return 2 * np.sum(a * np.array(x) + b - np.array(y))


def errors_func_ration(x: list, y: list, a: float, b: float) -> np.array:
    """

    """

    return np.sum(((a / (1 + b * np.array(x))) - np.array(y)) ** 2)


def derivative_a_errors_func_ration(x: list, y: list, a: float, b: float) -> np.array:
    return 2 * np.sum(((a / (1 + b * np.array(x))) - np.array(y)) * (1 / (1 + b * np.array(x))))


def derivative_b_errors_func_ration(x: list, y: list, a: float, b: float) -> np.array:
    return -2 * np.sum((a * np.array(x) * (a / (1 + b * np.array(x)) - np.array(y))) / (1 + b * np.array(x)) ** 2)


def gradient_descent(func, der_a, der_b, x, y, epsilon: float = 0.001, max_iter: int = 5000):
    approx = np.random.random(2)
    i = 0
    diff = np.inf
    f_old = func(x, y, approx[0], approx[1])

    while i < max_iter and diff > epsilon:
        grad = np.array([der_a(x, y, approx[0], approx[1]),
                         der_b(x, y, approx[0], approx[1])])

        lmbd = find_min_lambda(func, approx, grad, x, y)

        approx = approx + lmbd * grad

        f_new = func(x, y, approx[0], approx[1])

        diff = np.absolute(f_old - f_new)

        f_old = f_new

        i += 1

    return approx[0], approx[1]


def conjugate_gradient_descent(func, der_a, der_b, x, y, epsilon=0.001, max_iter=5000):
    i = 0
    approx = np.random.random(2)
    s = (-1) * np.array([der_a(x, y, approx[0], approx[1]),
                         der_b(x, y, approx[0], approx[1])])

    while i < max_iter and np.linalg.norm(s) > epsilon:
        lmbd = find_min_lambda(func, approx, s, x, y)

        grad_old = np.array([der_a(x, y, approx[0], approx[1]),
                             der_b(x, y, approx[0], approx[1])])

        approx = approx + lmbd * s

        grad_new = np.array([der_a(x, y, approx[0], approx[1]),
                             der_b(x, y, approx[0], approx[1])])

        w = np.linalg.norm(grad_new) ** 2 / np.linalg.norm(grad_old) ** 2

        s = (-1) * grad_new + w * s

        i += 1

    return approx[0], approx[1]






def find_min_lambda(func, approx, gradient, x, y):
    def minimized_function(params):
        lambd = params[0]

        return func(x, y, (approx + lambd * gradient)[0], (approx + lambd * gradient)[1])

    best = minimize(minimized_function, x0=np.array([0]), method='Nelder-Mead', tol=0.001)

    return best['x'][0]



# Creating of data
alpha = random.uniform(0.00000001, 0.99999999)
beta = random.uniform(0.00000001, 0.99999999)

x_list = [k / 100 for k in range(101)]
y_list = [alpha * xk + beta + random.normalvariate(0, 1) for xk in x_list]



plt.plot(x_list, y_list)
plt.plot(x_list, lin_func(x_list, *gradient_descent(errors_func_lin, derivative_a_errors_func_lin,
                                                    derivative_b_errors_func_lin,
                                                    x_list, y_list)), label="gradient descent linear",
         color='orange')
plt.plot(x_list, lin_func(x_list, *conjugate_gradient_descent(errors_func_lin, derivative_a_errors_func_ration,
                                                    derivative_b_errors_func_ration,
                                                    x_list, y_list)), label="gradient descent linear",
         color='red')

plt.show()


plt.plot(x_list, y_list)
plt.plot(x_list, ration_func(x_list, *gradient_descent(errors_func_ration, derivative_a_errors_func_ration,
                                                    derivative_b_errors_func_ration,
                                                    x_list, y_list)), label="gradient descent rational",
         color='orange')
plt.plot(x_list, ration_func(x_list, *conjugate_gradient_descent(errors_func_lin, derivative_a_errors_func_ration,
                                                    derivative_b_errors_func_ration,
                                                    x_list, y_list)), label="gradient descent linear",
         color='red')

plt.show()
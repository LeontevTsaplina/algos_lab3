import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Callable


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
    Function for finding errors function of linear function

    :param x: list of elements
    :param y: list of results
    :param a: coefficient a
    :param b: coefficient b
    :type x: list
    :type y: list
    :type a: coefficient a
    :type b: coefficient b
    :return: array of results of counting
    :rtype: np.array
    """

    return np.sum(((a * np.array(x) + b) - np.array(y)) ** 2)


def derivative_a_errors_func_lin(x: list, y: list, a: float, b: float) -> np.array:
    """
    Function for finding errors function of derivative of linear function by a coefficient

    :param x: list of elements
    :param y: list of results
    :param a: coefficient a
    :param b: coefficient b
    :type x: list
    :type y: list
    :type a: coefficient a
    :type b: coefficient b
    :return: array of results of counting
    :rtype: np.array
    """

    return 2 * np.sum((a * np.array(x) + b - np.array(y)) * np.array(x))


def derivative_b_errors_func_lin(x: list, y: list, a: float, b: float) -> np.array:
    """
    Function for finding errors function of derivative of linear function by b coefficient

    :param x: list of elements
    :param y: list of results
    :param a: coefficient a
    :param b: coefficient b
    :type x: list
    :type y: list
    :type a: coefficient a
    :type b: coefficient b
    :return: array of results of counting
    :rtype: np.array
    """

    return 2 * np.sum(a * np.array(x) + b - np.array(y))


def errors_func_ration(x: list, y: list, a: float, b: float) -> np.array:
    """
    Function for finding errors function of rational function

    :param x: list of elements
    :param y: list of results
    :param a: coefficient a
    :param b: coefficient b
    :type x: list
    :type y: list
    :type a: coefficient a
    :type b: coefficient b
    :return: array of results of counting
    :rtype: np.array
    """

    return np.sum(((a / (1 + b * np.array(x))) - np.array(y)) ** 2)


def derivative_a_errors_func_ration(x: list, y: list, a: float, b: float) -> np.array:
    """
    Function for finding errors function of derivative of rational function by a coefficient

    :param x: list of elements
    :param y: list of results
    :param a: coefficient a
    :param b: coefficient b
    :type x: list
    :type y: list
    :type a: coefficient a
    :type b: coefficient b
    :return: array of results of counting
    :rtype: np.array
    """

    return 2 * np.sum(((a / (1 + b * np.array(x))) - np.array(y)) * (1 / (1 + b * np.array(x))))


def derivative_b_errors_func_ration(x: list, y: list, a: float, b: float) -> np.array:
    """
    Function for finding errors function of derivative of rational function by b coefficient

    :param x: list of elements
    :param y: list of results
    :param a: coefficient a
    :param b: coefficient b
    :type x: list
    :type y: list
    :type a: coefficient a
    :type b: coefficient b
    :return: array of results of counting
    :rtype: np.array
    """

    return -2 * np.sum((a * np.array(x) * (a / (1 + b * np.array(x)) - np.array(y))) / (1 + b * np.array(x)) ** 2)


def gradient_descent(func: Callable, der_a: Callable, der_b: Callable, x: list, y: list,
                     epsilon: float = 0.001, max_iter: int = 5000) -> tuple:
    """
    Function of gradient descent to find minimum of input function

    :param func: input function
    :param der_a: derivative by a coefficient of input function
    :param der_b: derivative by b coefficient of input function
    :param x: list of elements
    :param y: list of results
    :param epsilon: epsilon
    :param max_iter: maximum count of iterations
    :type: func: Callable
    :type der_a: Callable
    :type der_b: Callable
    :type x: list
    :type y: list
    :type epsilon: float
    :tupe max_iter: int
    :return: the best a and b coefficients
    :rtype: tuple
    """

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


def conjugate_gradient_descent(func: Callable, der_a: Callable, der_b: Callable, x: list, y: list,
                               epsilon: float = 0.001, max_iter: int = 5000):
    """
    Function of conjugate gradient descent to find minimum of input function

    :param func: input function
    :param der_a: derivative by a coefficient of input function
    :param der_b: derivative by b coefficient of input function
    :param x: list of elements
    :param y: list of results
    :param epsilon: epsilon
    :param max_iter: maximum count of iterations
    :type: func: Callable
    :type der_a: Callable
    :type der_b: Callable
    :type x: list
    :type y: list
    :type epsilon: float
    :tupe max_iter: int
    :return: the best a and b coefficients
    :rtype: tuple
    """

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


def newton(func: Callable, der: Callable) -> tuple:
    """
    Function of newton's method to find minimum of input function

    :param func: input function
    :param der: gradient of function
    :type: func: Callable
    :type der: Callable
    :return: the best a and b coefficients
    :rtype: tuple
    """

    res = minimize(func, np.array([0, 0]), method='Newton-CG',
                   jac=der,
                   options={'xtol': 1e-8, 'disp': True})

    return res['x'][0], res['x'][1]


def levenberg_marquardt(func: Callable) -> tuple:
    """
    Function of levenberg-marquardt's method to find minimum of input function

    :param func: input function
    :type: func: Callable
    :return: the best a and b coefficients
    :rtype: tuple
    """

    res = least_squares(func, np.array([0, 0]), method='lm')

    return res['x'][0], res['x'][1]


def find_min_lambda(func: Callable, approx: list, gradient: list, x: list, y: list) -> float:
    """
    Function to find the best lambda coefficient for input function by Nelder-Mead's method
    of one-dimension minimization

    :param func: input function
    :param approx: coefficients
    :param gradient: gradient of function
    :param x: list of elements
    :param y: list of results
    :type func: Callable
    :type approx: list
    :type gradient: list
    :type x: list
    :type y: list
    :return: best lambda
    :rtype: float
    """

    def minimized_function(params: list) -> float:
        """
        Result of minimized function

        :param params: list of coefficients
        :type params: list
        :return: result of function
        :rtype: float
        """

        lambd = params[0]

        return func(x, y, (approx + lambd * gradient)[0], (approx + lambd * gradient)[1])

    best = minimize(minimized_function, x0=np.array([0]), method='Nelder-Mead', tol=0.001)

    return best['x'][0]


# Creating of data
alpha = random.uniform(0.00000001, 0.99999999)
beta = random.uniform(0.00000001, 0.99999999)

x_list = [k / 100 for k in range(101)]
y_list = [alpha * xk + beta + random.normalvariate(0, 1) for xk in x_list]


def errors_func_lin_lev_marq(params: list) -> list:
    """
    Function for finding errors function of linear function for levenberg marquardt's method
    (because of specific of scipy)

    :param params: params a and b
    :type params: list
    :return: list [errors sum, errors sum]
    :rtype: list
    """

    a = params[0]
    b = params[1]
    return [np.sum((a * np.array(x_list) + b - np.array(y_list)) ** 2),
            np.sum((a * np.array(x_list) + b - np.array(y_list)) ** 2)]


def errors_func_lin_lev_marq_newton(params: list) -> np.array:
    """
    Function for finding errors function of linear function for levenberg marquardt's method for newton's method
    (because of specific of scipy)

    :param params: params a and b
    :type params: list
    :return: errors sum
    :rtype: np.array
    """

    a = params[0]
    b = params[1]
    return np.sum((a * np.array(x_list) + b - np.array(y_list)) ** 2)


def errors_func_lin_lev_marq_der(params: list) -> np.array:
    """
    Function for finding gradient of errors function of linear function for levenberg marquardt's method
    (because of specify of scipy)

    :param params: list of params
    :type params: list
    :return: gradient of linear function
    :rtype: np.array
    """

    a = params[0]
    b = params[1]

    der = np.array([
        derivative_a_errors_func_lin(x_list, y_list, a, b),
        derivative_b_errors_func_lin(x_list, y_list, a, b)
    ])

    return der


def errors_func_ration_lev_marq(params: list) -> list:
    """
    Function for finding errors function of rational function for levenberg marquardt's method
    (because of specific of scipy)

    :param params: params a and b
    :type params: list
    :return: [errors sum, errors sum]
    :rtype: list
    """

    a = params[0]
    b = params[1]
    return [np.sum(((a / (1 + b * np.array(x_list))) - np.array(y_list)) ** 2),
            np.sum(((a / (1 + b * np.array(x_list))) - np.array(y_list)) ** 2)]


def errors_func_ration_lev_marq_newton(params: list) -> np.array:
    """
    Function for finding errors function of rational function for levenberg marquardt's method for newton's method
    (because of specific of scipy)

    :param params: params a and b
    :type params: list
    :return: errors sum
    :rtype: np.array
    """

    a = params[0]
    b = params[1]
    return np.sum(((a / (1 + b * np.array(x_list))) - np.array(y_list)) ** 2)


def errors_func_ration_lev_marq_der(params: list):
    """
    Function for finding gradient of errors function of rational function for levenberg marquardt's method
    (because of specify of scipy)

    :param params: list of params
    :type params: list
    :return: gradient of rational function
    :rtype: np.array
    """

    a = params[0]
    b = params[1]

    der = np.array([
        derivative_a_errors_func_ration(x_list, y_list, a, b),
        derivative_b_errors_func_ration(x_list, y_list, a, b)
    ])

    return der


# Plot for approximation by linear function
plt.plot(x_list, y_list)
plt.plot(x_list, lin_func(x_list, *gradient_descent(errors_func_lin, derivative_a_errors_func_lin,
                                                    derivative_b_errors_func_lin,
                                                    x_list, y_list)), label="gradient descent linear",
         color='orange')
plt.plot(x_list, lin_func(x_list, *conjugate_gradient_descent(errors_func_lin, derivative_a_errors_func_lin,
                                                              derivative_b_errors_func_lin, x_list, y_list)),
         label="conjugate gradient descent linear", color='red')
plt.plot(x_list, lin_func(x_list, *newton(errors_func_lin_lev_marq_newton, errors_func_lin_lev_marq_der)),
         label="newton linear", color='green')
plt.plot(x_list, lin_func(x_list, *levenberg_marquardt(errors_func_lin_lev_marq)),
         label="levenberg marquardt linear", color='purple')
plt.legend()
plt.show()

# Plot for approximation by rational function
plt.plot(x_list, y_list)
plt.plot(x_list, ration_func(x_list, *gradient_descent(errors_func_ration, derivative_a_errors_func_ration,
                                                       derivative_b_errors_func_ration,
                                                       x_list, y_list)), label="gradient descent ration",
         color='orange')
plt.plot(x_list, ration_func(x_list, *conjugate_gradient_descent(errors_func_ration, derivative_a_errors_func_ration,
                                                                 derivative_b_errors_func_ration, x_list, y_list)),
         label="conjugate gradient descent ration", color='red')
plt.plot(x_list, ration_func(x_list, *newton(errors_func_ration_lev_marq_newton, errors_func_ration_lev_marq_der)),
         label="newton ration", color='green')
plt.plot(x_list, ration_func(x_list, *levenberg_marquardt(errors_func_ration_lev_marq)),
         label="levenberg marquardt ration", color='purple')

plt.legend()
plt.show()

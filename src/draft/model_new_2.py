import numpy as np
import matplotlib.pyplot as plt
import math, copy

def compute_cost(x, y, w, b):
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

x_train = np.array([1.0, 2.0, 4.0])
y_train = np.array([300.0, 500.0, 1000.0])

plt.scatter(x_train, y_train)
plt.show()

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]

        dj_dw = dj_dw + dj_dw_i
        dj_db = dj_db + dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # update param
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 10000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        if i% math.ceil(num_iters / 10) == 0:
            print(f'Iteration {i:4d}: Cost {J_history[-1]:0.2e}   dw: {dj_dw:0.3e}   db: {dj_db:0.3e}  w: {w:0.3e}   b: {b:0.3e}')

    return w, b, J_history, p_history # return final w, b and J history for graphing

# initial w, b, and learning rate
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 1000
tmp_alpha = 0.01

# run gradient descent
w_final, b_final, J_history, p_history = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
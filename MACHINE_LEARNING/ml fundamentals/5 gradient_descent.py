import numpy as np
import matplotlib.pyplot as plt
import math

# Training data
x = np.array([1, 2])
y = np.array([400, 600])

# Cost Function
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2

    total_cost = cost / (2 * m)
    return total_cost


# Gradient Function
def compute_gradient(x, y, w, b):
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b

        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])

        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


# Gradient Descent
def gradient_descent(x, y, w_in, b_in,
                     alpha, num_iters,
                     cost_function,
                     gradient_function):

    J_history = []
    P_history = []

    w = w_in
    b = b_in

    for i in range(num_iters):

        # Compute gradients
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost and parameters
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            P_history.append([w, b])

        # Print progress ~10 times
        if i % math.ceil(num_iters / 10) == 0:
            print(
                f"Iteration {i:5}: "
                f"Cost {J_history[-1]:0.2e} "
                f"dj_dw: {dj_dw:0.3e}, "
                f"dj_db: {dj_db:0.3e}, "
                f"w: {w:0.3e}, "
                f"b: {b:0.3e}"
            )

    return w, b, J_history, P_history


# Initial Parameters
w_init = 0
b_init = 0

# Hyperparameters
num_iters = 10000
alpha = 1.0e-2

# Run Gradient Descent
w_final, b_final, J_hist, P_hist = gradient_descent(
    x,
    y,
    w_init,
    b_init,
    alpha,
    num_iters,
    compute_cost,
    compute_gradient
)

print(f"\n(w,b) found by gradient descent: ({w_final:.4f}, {b_final:.4f})")

# Plot Training Data and Learned Line
plt.figure(figsize=(6, 4))

plt.scatter(
    x,
    y,
    marker='x',
    color='red',
    s=100,
    label='Training Data'
)

f_wb = w_final * x + b_final

plt.plot(
    x,
    f_wb,
    color='blue',
    label='Prediction'
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

# Plot Cost History
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    constrained_layout=True,
    figsize=(12, 4)
)

ax1.plot(J_hist[:100])
ax1.set_title("Cost vs Iteration (Start)")
ax1.set_ylabel("Cost")
ax1.set_xlabel("Iteration")

ax2.plot(
    1000 + np.arange(len(J_hist[1000:])),
    J_hist[1000:]
)
ax2.set_title("Cost vs Iteration (End)")
ax2.set_ylabel("Cost")
ax2.set_xlabel("Iteration")

plt.show()
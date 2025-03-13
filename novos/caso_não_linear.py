import numpy as np
import matplotlib.pyplot as plt

# Define the nonlinear function (quadratic)
def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c

# Define partial derivatives for gradient descent
def derivatives(x, y, a, b, c):
    y_pred = quadratic_function(x, a, b, c)
    da = -2 * np.sum(x**2 * (y - y_pred))  # partial derivative w.r.t. a
    db = -2 * np.sum(x * (y - y_pred))     # partial derivative w.r.t. b
    dc = -2 * np.sum(y - y_pred)           # partial derivative w.r.t. c
    return da, db, dc

# Data from the table
x_data = np.array([-1.0, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.0])
y_data = np.array([36.547, 17.264, 8.155, 3.852, 1.820, 0.860, 0.406, 0.246])

# Invert the data
inverted_x = y_data
inverted_y = x_data

# Compute Newton's divided difference coefficients for the inverted data
def newton_divided_difference(x_points, y_points):
    n = len(x_points)
    F = np.zeros((n, n))
    F[:, 0] = y_points  # First column is y values
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x_points[i + j] - x_points[i])
    return F[0, :]  # Return the top row (coefficients)

coeffs = newton_divided_difference(inverted_x, inverted_y)

# Define the Newton polynomial for the inverted data
def newton_polynomial(x_points, coeffs, x):
    n = len(x_points) - 1
    result = coeffs[n]
    for i in range(n - 1, -1, -1):
        result = result * (x - x_points[i]) + coeffs[i]
    return result

# Solve for x when y = 0.5
y_target = 0.7
x_guess = 0.0  # Initial guess
tolerance = 1e-8
max_iterations = 10000
iteration = 0

while iteration < max_iterations:
    y_value = newton_polynomial(inverted_x, coeffs, x_guess)
    error = y_value - y_target
    if abs(error) < tolerance:
        break
    # Compute the derivative of the Newton polynomial
    derivative = sum(coeffs[i] * np.prod([x_guess - inverted_x[j] for j in range(i)]) for i in range(len(coeffs)))
    x_guess -= error / derivative  # Newton's method update
    iteration += 1

if iteration < max_iterations:
    print(f"Found x = {x_guess:.6f} for y = {y_target}")
else:
    print("Solution did not converge")

# Plotting
x_dense = np.linspace(min(inverted_x), max(inverted_x), 200)
y_dense = np.array([newton_polynomial(inverted_x, coeffs, x) for x in x_dense])

plt.figure(figsize=(10, 6))
plt.scatter(inverted_x, inverted_y, color='blue', label='Data points')
plt.plot(x_dense, y_dense, color='red', label='Newton Polynomial')
plt.xlabel('y')
plt.ylabel('x')
plt.title('Inverse Interpolation using Newton\'s Divided Difference')
plt.legend()
plt.grid(True)
plt.show()

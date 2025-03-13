import numpy as np
import matplotlib.pyplot as plt

def cubic_spline_interpolation(x_data, y_data, num_points=100):
    """
    Calculate and plot cubic spline interpolation, show R² and piecewise functions.
    Uses natural cubic spline method where second derivatives at endpoints are zero.
    
    Parameters:
    x_data : array-like, x coordinates of data points (must be in ascending order)
    y_data : array-like, y coordinates of data points (same length as x_data)
    num_points : int, number of points to generate in the interpolated curve
    
    Returns:
    None (displays plot and prints R² and functions)
    """
    
    # Convert input lists to numpy arrays
    x = np.array(x_data)
    y = np.array(y_data)
    n = len(x) - 1  # Number of intervals
    h = np.diff(x)  # Interval lengths
    
    # Set up and solve tridiagonal system
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    A[0, 0] = 1  # Natural spline boundary
    A[n, n] = 1  # Natural spline boundary
    
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3 * ((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    
    c = np.linalg.solve(A, b)  # Second derivatives
    
    # Calculate coefficients
    a = y[:-1]
    b_coef = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        b_coef[i] = (y[i+1] - y[i])/h[i] - h[i]*(c[i+1] + 2*c[i])/3
        d[i] = (c[i+1] - c[i])/(3*h[i])
    
    # Generate interpolated points
    x_new = np.linspace(x[0], x[-1], num_points)
    y_new = np.zeros(num_points)
    
    for i in range(num_points):
        for j in range(n):
            if x[j] <= x_new[i] <= x[j+1]:
                t = x_new[i] - x[j]
                y_new[i] = a[j] + b_coef[j]*t + c[j]*t**2 + d[j]*t**3
                break
    
    # Calculate R²
    # Interpolate at original x points for comparison
    y_interp = np.zeros_like(y)
    for i in range(len(x)):
        for j in range(n):
            if x[j] <= x[i] <= x[j+1]:
                t = x[i] - x[j]
                y_interp[i] = a[j] + b_coef[j]*t + c[j]*t**2 + d[j]*t**3
                break
    
    # R² calculation
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)  # Total sum of squares
    ss_res = np.sum((y - y_interp)**2)  # Residual sum of squares
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    # Print piecewise functions
    print("Piecewise Cubic Functions (where t = x - x_start):")
    for i in range(n):
        func = f"S_{i}(t) = {a[i]:.4f}"
        if b_coef[i] != 0:
            func += f" + {b_coef[i]:.4f}t"
        if c[i] != 0:
            func += f" + {c[i]:.4f}t²"
        if d[i] != 0:
            func += f" + {d[i]:.4f}t³"
        print(f"For {x[i]:.2f} ≤ x ≤ {x[i+1]:.2f}: {func}")
    
    print(f"\nR² = {r_squared:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro', label='Data points')
    plt.plot(x_new, y_new, 'b-', label='Cubic spline')
    plt.grid(True)
    plt.legend()
    plt.title(f'Cubic Spline Interpolation (R² = {r_squared:.4f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Example usage
if __name__ == "__main__":
    x_data = [0, 1, 2, 3, 4]
    y_data = [0, 1, 0, 1, 0]
    cubic_spline_interpolation(x_data, y_data)
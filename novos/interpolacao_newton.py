import numpy as np
import matplotlib.pyplot as plt

def newton_divided_difference(x_points, y_points):
    """
    Compute divided differences for Newton's interpolation
    Returns the coefficients of the Newton polynomial
    """
    n = len(x_points)
    # Create divided difference table
    F = np.zeros((n, n))
    F[:, 0] = y_points  # First column is y values
    
    # Fill the divided difference table
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x_points[i + j] - x_points[i])
    #tabela de direfenças divididas --> delta(Y)/delta(Y)
    return F[0, :]  # Return the top row (coefficients)

def newton_polynomial(x_points, coeffs, x):
    """
    Evaluate the Newton polynomial at point x
    x_points: array of x coordinates
    coeffs: divided difference coefficients
    x: point where to evaluate
    """
    n = len(x_points) - 1
    result = coeffs[n]  # Start with highest-order coefficient
    
    # Horner's method for polynomial evaluation
    for i in range(n - 1, -1, -1):
        result = result * (x - x_points[i]) + coeffs[i]
    
    return result

def get_user_points():
    """
    Collect points from user input until an empty line is entered
    Returns numpy arrays of x and y coordinates
    """
    x_points = []
    y_points = []
    
    print("Enter points as 'x,y' (e.g., '1,2'). Press Enter twice to finish:")
    
    while True:
        point = input("Enter point (x,y): ").strip()
        if point == "":
            if len(x_points) < 2:
                print("Need at least 2 points for interpolation. Please add more.")
                continue
            break
        
        try:
            x, y = map(float, point.split(','))
            if x in x_points:
                print("Error: Duplicate x value. All x coordinates must be unique.")
                continue
            x_points.append(x)
            y_points.append(y)
            print(f"Added point: ({x}, {y})")
        except ValueError:
            print("Invalid format. Please enter as 'x,y' using numbers.")
    
    return np.array(x_points), np.array(y_points)

if __name__ == "__main__":
    # Get points from user
    x_points, y_points = get_user_points()
    
    # Compute Newton's divided difference coefficients
    coeffs = newton_divided_difference(x_points, y_points)
    
    # Create dense x values for smooth curve
    x_dense = np.linspace(min(x_points), max(x_points), 200)
    
    # Calculate interpolated values
    y_dense = np.array([newton_polynomial(x_points, coeffs, x) for x in x_dense])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot original points
    plt.scatter(x_points, y_points, color='red', label='Given Points', zorder=5)
    
    # Plot interpolated curve
    plt.plot(x_dense, y_dense, 'b-', label='Newton Polynomial')
    
    # Customize plot
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Newton Interpolation Polynomial')
    plt.legend()
    
    # Print polynomial coefficients
    print("\nNewton divided difference coefficients:")
    for i, coef in enumerate(coeffs):
        indices = ','.join(str(j) for j in range(i + 1))
        print(f"f[{indices}] = {coef:.6f}")
    
    # Compare with standard polynomial form
    poly_coeffs = np.polyfit(x_points, y_points, len(x_points)-1)
    polynomial = np.poly1d(poly_coeffs)
    print("\nEquivalent polynomial coefficients (highest degree first):")
    print(polynomial)
    
    plt.show()

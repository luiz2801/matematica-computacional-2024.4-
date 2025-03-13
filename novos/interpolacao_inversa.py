import numpy as np
import matplotlib.pyplot as plt

def newton_divided_difference(y_points, x_points):
    """
    Compute divided differences for Newton's interpolation
    Returns the coefficients of the Newton polynomial
    """
    n = len(y_points)
    # Create divided difference table
    F = np.zeros((n, n))
    F[:, 0] = x_points  # First column is x values
    
    # Fill the divided difference table
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (y_points[i + j] - y_points[i])
    
    return F[0, :]  # Return the top row (coefficients)

def newton_polynomial(y_points, coeffs, y):
    """
    Evaluate the Newton polynomial at point y
    y_points: array of y coordinates
    coeffs: divided difference coefficients
    y: point where to evaluate
    """
    n = len(y_points) - 1
    result = coeffs[n]  # Start with highest-order coefficient
    
    # Horner's method for polynomial evaluation
    for i in range(n - 1, -1, -1):
        result = result * (y - y_points[i]) + coeffs[i]
    
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
            if y in y_points:
                print("Error: Duplicate y value. All y coordinates must be unique.")
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
    
    # Compute Newton's divided difference coefficients for inverse interpolation
    coeffs = newton_divided_difference(y_points, x_points)
    
    # Create dense y values for smooth curve
    y_dense = np.linspace(min(y_points), max(y_points), 200)
    
    # Calculate interpolated x values
    x_dense = np.array([newton_polynomial(y_points, coeffs, y) for y in y_dense])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot original points
    plt.scatter(x_points, y_points, color='red', label='Given Points', zorder=5)
    
    # Plot interpolated curve
    plt.plot(x_dense, y_dense, 'b-', label='Inverse Newton Polynomial')
    
    # Customize plot
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Inverse Newton Interpolation')
    plt.legend()
    
    # Print polynomial coefficients
    print("\nInverse Newton divided difference coefficients:")
    for i, coef in enumerate(coeffs):
        indices = ','.join(str(j) for j in range(i + 1))
        print(f"f[{indices}] = {coef:.6f}")
    
    plt.show()

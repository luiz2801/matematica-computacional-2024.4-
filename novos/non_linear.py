import numpy as np

def newton_method(F, J, x0, tol=1e-8, max_iter=10000):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        f_val = F(x)
        J_val = J(x)
        
        # Check convergence
        if np.linalg.norm(f_val) < tol:
            print(f"Converged in {i} iterations")
            return x
        
        # Check if Jacobian is singular
        det = np.linalg.det(J_val)
        print(f"Iteration {i}: x = {x}, det(J) = {det}, F(x) = {f_val}")
        
        if abs(det) < 1e-10:  # If Jacobian is nearly singular
            print("Warning: Jacobian is singular or nearly singular")
            return None
        
        try:
            # Solve J * delta = -F
            delta = np.linalg.solve(J_val, -f_val)
            x = x + delta
        except np.linalg.LinAlgError:
            print("Error: Singular matrix encountered")
            return None
            
    print("Warning: Maximum iterations reached")
    return x

# Corrected example system with independent equations
def F_example(x):
    return np.array([
        2*x[0] + 3*x[1] + x[2] -11,
        x[0] - x[1] + x[2] - 6,
        5*x[0] + 2*x[1] + 3*x[2] - 18
    ])

def J_example(x):
    return np.array([
        [2, 3, 1],
        [1, 1, 1],
        [5, 2, 3]
    ])

if __name__ == "__main__":
    x0 = [0.0, 0.0, 0.0]  # Initial guess
    
    solution = newton_method(F_example, J_example, x0)
    
    if solution is not None:
        print(f"\nFinal solution: x = {solution[0]:.6f}, y = {solution[1]:.6f}, z = {solution[2]:.6f}")
        print(f"Verification: F(solution) = {F_example(solution)}")
    else:
        print("Failed to find a solution. Try a different initial guess.")

import numpy as np

def simpson_one_third(func, a, b, n):
    """
    Calculate definite integral using Simpson's 1/3 rule.
    
    Parameters:
    func : callable - function to integrate (must take one argument)
    a : float - lower bound of integration
    b : float - upper bound of integration
    n : int - number of intervals (must be even)
    
    Returns:
    float - approximate value of the definite integral
    """
    
    if n % 2 != 0:
        raise ValueError("Number of intervals (n) must be even for Simpson's 1/3 rule")
    
    h = (b - a) / n  # Step size
    
    # Compute function values directly to avoid storing large arrays
    y0 = func(a)
    yn = func(b)
    odd_sum = sum(func(a + i * h) for i in range(1, n, 2))  # Odd indices (4x)
    even_sum = sum(func(a + i * h) for i in range(2, n, 2)) # Even indices (2x)
    
    integral = (h / 3) * (y0 + yn + 4 * odd_sum + 2 * even_sum)
    
    return integral

# Example usage
def test_function(x):
    return np.exp(x)  # f(x) = x*sin(x)

if __name__ == "__main__":
    a = 1  # Lower bound
    b = 2  # Upper bound
    n =  4   # Number of intervals (must be even)
    
    result = simpson_one_third(test_function, a, b, n)
    exact_result = np.sin(1) - np.cos(1)  # Correct result
    error = abs(result - exact_result)

    print(f"Numerical integration result: {result}")
    print(f"Exact result: {exact_result}")
    print(f"Absolute error: {error}")

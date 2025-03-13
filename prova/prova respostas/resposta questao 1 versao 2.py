import numpy as np
import matplotlib.pyplot as plt

def runge_function(x):
    """
    Define a função de Runge f(x) = 1 / (1 + 25x^2).
    """
    return 1 / (1 + 25 * x**2)

def lagrange_interpolation(x_values, y_values, x_interp):
    """
    Implementação da interpolação de Lagrange.
    """
    n = len(x_values)
    result = 0
    
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x_interp - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    
    return result

def cubic_spline_interpolation(x_values, y_values, x_interp):
    """
    Implementação da interpolação spline cúbica manualmente.
    """
    n = len(x_values) - 1
    a = y_values
    b = np.zeros(n)
    c = np.zeros(n + 1)
    d = np.zeros(n)
    h = [x_values[i + 1] - x_values[i] for i in range(n)]
    
    # Construção do sistema linear para encontrar c
    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)
    A[0][0] = 1  # Condição de contorno natural
    A[n][n] = 1
    
    for i in range(1, n):
        A[i][i - 1] = h[i - 1]
        A[i][i] = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]
        B[i] = (3 / h[i]) * (a[i + 1] - a[i]) - (3 / h[i - 1]) * (a[i] - a[i - 1])
    
    c = np.linalg.solve(A, B)
    
    # Cálculo dos coeficientes b e d
    for i in range(n):
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    
    # Interpolação
    for i in range(n):
        if x_values[i] <= x_interp <= x_values[i + 1]:
            dx = x_interp - x_values[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    
    raise ValueError("x_interp está fora do intervalo fornecido.")

# Definição do intervalo e número de pontos
x_real = np.linspace(-2, 2, 400)
y_real = runge_function(x_real)

# Diferentes amostragens
samples = [6, 10, 14]
interpolations = {}

plt.figure(figsize=(10, 6))
plt.plot(x_real, y_real, label='Função Real', color='black', linestyle='dashed')

for num_points in samples:
    x_samples = np.linspace(-2, 2, num_points)
    y_samples = runge_function(x_samples)
    y_interp = [lagrange_interpolation(x_samples, y_samples, x) for x in x_real]
    plt.plot(x_real, y_interp, label=f'Interpolação {num_points} pontos')
    interpolations[num_points] = (x_samples, y_samples)

# Aplicando a spline cúbica na amostra de 10 pontos
x_spline, y_spline = interpolations[10]
y_spline_interp = [cubic_spline_interpolation(x_spline, y_spline, x) for x in x_real]
plt.plot(x_real, y_spline_interp, label='Spline Cúbica (10 pontos)', linestyle='dotted')

plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Interpolação da Função de Runge')
plt.grid()
plt.show()

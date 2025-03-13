import numpy as np
import matplotlib.pyplot as plt

def runge_function(x):
    """
    Função de Runge, f(x) = 1 / (1 + 25x^2)
    """
    return 1 / (1 + 25 * x**4)

def cubic_spline_interpolation(x_vals, y_vals, x_interp):
    """
    Função para interpolação spline cúbica natural.
    """
    n = len(x_vals) - 1
    h = np.diff(x_vals)  # Diferenças entre pontos consecutivos
    alpha = np.zeros(n)
    l = np.ones(n + 1)
    mu = np.zeros(n)
    z = np.zeros(n + 1)

    # Preenchendo a matriz do sistema
    for i in range(1, n):
        alpha[i] = (3 / h[i-1]) * (y_vals[i] - y_vals[i-1]) - (3 / h[i]) * (y_vals[i+1] - y_vals[i])

    # Resolver o sistema tridiagonal
    for i in range(1, n):
        l[i] = 2 * (x_vals[i+1] - x_vals[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    l[n] = 1
    z[n] = 0
    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)

    # Calculando os coeficientes c, b, d
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y_vals[j+1] - y_vals[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])

    # Calculando a interpolação para os valores de x_interp
    for i in range(n):
        if x_vals[i] <= x_interp <= x_vals[i+1]:
            dx = x_interp - x_vals[i]
            return y_vals[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

    return None

# Geração dos pontos
num_points = [6, 10, 14]
x_vals_fine = np.linspace(-2, 2, 500)
y_true = runge_function(x_vals_fine)

# Plotando a função real de Runge
plt.figure(figsize=(10, 6))
plt.plot(x_vals_fine, y_true, 'k-', label='Função de Runge (Real)')

# Realizando as interpolações
for n in num_points:
    x_sample = np.linspace(-2, 2, n)
    y_sample = runge_function(x_sample)
    
    # Interpolação Polinomial (usando BarycentricInterpolator, como em seu exemplo)
    y_poly = np.zeros_like(x_vals_fine)
    for i, x in enumerate(x_vals_fine):
        y_poly[i] = cubic_spline_interpolation(x_sample, y_sample, x)
    
    # Plotando a interpolação
    plt.plot(x_vals_fine, y_poly, '--', label=f'Interpolação Polinomial ({n} pontos)')
    plt.scatter(x_sample, y_sample, marker='o', label=f'Amostras ({n} pontos)')

# Interpolação Spline Cúbica (usando amostras de 10 pontos)
x_spline = np.linspace(-2, 2, 10)
y_spline = runge_function(x_spline)

# Calculando a spline cúbica para 10 pontos
y_spline_vals = np.zeros_like(x_vals_fine)
for i, x in enumerate(x_vals_fine):
    y_spline_vals[i] = cubic_spline_interpolation(x_spline, y_spline, x)

# Plotando a spline cúbica
plt.plot(x_vals_fine, y_spline_vals, '-.', label='Spline Cúbica (10 pontos)')

# Configuração do gráfico
plt.title("Interpolação de Função de Runge")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

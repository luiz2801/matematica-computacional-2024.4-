import numpy as np
import matplotlib.pyplot as plt
from spline import quadratic_spline_interpolation

# Definição da função de Runge
def runge_function(x):
    return 1 / (1 + 25 * x**2)

# Geração de amostras igualmente espaçadas para os três conjuntos
num_points = [6, 10, 14]  # Quantidade de pontos nas amostras
x_vals_fine = np.linspace(-2, 2, 500)  # Valores para plotagem suave da função

y_true = runge_function(x_vals_fine)  # Valores reais da função

plt.figure(figsize=(10, 6))
plt.plot(x_vals_fine, y_true, 'k-', label='Função de Runge (Real)')  # Função original

for n in num_points:
    x_sample = np.linspace(-2, 2, n)  # Pontos amostrados
    y_sample = runge_function(x_sample)
    
    # Interpolação polinomial de Lagrange
    poly_interp = quadratic_spline_interpolation(x_sample, y_sample, x_interp=3)
    y_poly = poly_interp(x_vals_fine)
    
    plt.plot(x_vals_fine, y_poly, '--', label=f'Interpolação Polinomial ({n} pontos)')
    plt.scatter(x_sample, y_sample, marker='o', label=f'Amostras ({n} pontos)')

# Interpolação Spline cúbica para a amostra de 10 pontos
x_spline = np.linspace(-2, 2, 10)  # 10 pontos
y_spline = runge_function(x_spline)
cs = CubicSpline(x_spline, y_spline, bc_type='natural')  # Criando spline cúbica

y_spline_vals = cs(x_vals_fine)  # Avaliação da spline nos pontos finos
plt.plot(x_vals_fine, y_spline_vals, '-.', label='Spline Cúbica (10 pontos)')

# Personalização do gráfico
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Interpolação Polinomial e Spline da Função de Runge')
plt.legend()
plt.grid()
plt.show()

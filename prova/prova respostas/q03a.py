import numpy as np
import matplotlib.pyplot as plt
import random

def lagrange_interpolation(points, x_interp=None):
    def L(k, x_vals):
        term = np.ones_like(x_vals, dtype=float)
        x_k, _ = points[k]
        for j, (x_j, _) in enumerate(points):
            if j != k:
                term *= (x_vals - x_j) / (x_k - x_j)
        return term
    
    def P(x_vals):
        return sum(y_k * L(k, x_vals) for k, (_, y_k) in enumerate(points))
    
    # Exibir a função interpoladora de forma simplificada
    coef = np.polyfit([p[0] for p in points], [p[1] for p in points], len(points) - 1)
    poly_eq = " + ".join(f"{c:.4f}x^{len(coef)-i-1}" for i, c in enumerate(coef) if abs(c) > 1e-6)
    print("Função interpoladora de Lagrange simplificada:")
    print(f"P(x) = {poly_eq}")
    
    x_vals = np.linspace(min(p[0] for p in points) - 1, max(p[0] for p in points) + 1, 400)
    y_vals = P(x_vals)
    
    plt.plot(x_vals, y_vals, label="Interpolação de Lagrange")
    plt.scatter(*zip(*points), color='red', label="Pontos dados")
    
    if x_interp is not None:
        y_interp = P(np.array([x_interp]))[0]
        plt.axvline(x_interp, color='g', linestyle="--", label=f"x_interp = {x_interp}")
        plt.scatter(x_interp, y_interp, color='purple', zorder=3, label=f"Interpolado: y ≈ {y_interp:.2f}")
        print(f"Valor interpolado em x = {x_interp}: y ≈ {y_interp:.4f}")
    
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.title("Interpolação de Lagrange")
    plt.grid()
    plt.show()

# Gerar pontos aleatórios
num_pontos = 100  # Número de pontos a serem gerados
x_vals = np.random.uniform(0, 20, num_pontos)  # Valores aleatórios de x entre 0 e 20
pontos = []

for x in x_vals:
    a = random.uniform(-5, 5)  # Valor aleatório para 'a' entre -5 e 5
    y = np.exp(x) + a  # Calcula f(x) = e^x + a
    pontos.append((x, y))

# Exibir os pontos gerados
print("Pontos gerados (x, y):")
for x, y in pontos:
    print(f"x = {x:.4f}, y = {y:.4f}")

# Realizar a interpolação
x_interp = float(input("Digite o valor de x para interpolação: "))
lagrange_interpolation(pontos, x_interp)

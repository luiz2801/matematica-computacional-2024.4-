import numpy as np
import matplotlib.pyplot as plt

# Dados fornecidos
x = np.array([0, 0.5, 1, 1.5, 2, 25])
y = np.array([-2.78, -2.241, -1.65, -0.594, 1.34, 4.564])

# Polinômios com diferentes graus usando os pontos necessários
# Linear (2 pontos: x = 0, 0.5)
p1 = np.polyfit(x[:2], y[:2], 1)
P1 = np.poly1d(p1)
print("Polinômio linear: P1(x) = {:.4f}x + {:.4f}".format(p1[0], p1[1]))

# Quadrático (3 pontos: x = 0, 0.5, 1)
p2 = np.polyfit(x[:3], y[:3], 2)
P2 = np.poly1d(p2)
print("Polinômio quadrático: P2(x) = {:.4f}x^2 + {:.4f}x + {:.4f}".format(p2[0], p2[1], p2[2]))

# Cúbico (4 pontos: x = 0, 0.5, 1, 1.5)
p3 = np.polyfit(x[:4], y[:4], 3)
P3 = np.poly1d(p3)
print("Polinômio cúbico: P3(x) = {:.4f}x^3 + {:.4f}x^2 + {:.4f}x + {:.4f}".format(p3[0], p3[1], p3[2], p3[3]))

# Calcular f(1.23) para cada polinômio
f_123_p1 = P1(1.23)
f_123_p2 = P2(1.23)
f_123_p3 = P3(1.23)
print(f"\nf(1.23) com linear: {f_123_p1:.4f}")
print(f"f(1.23) com quadrático: {f_123_p2:.4f}")
print(f"f(1.23) com cúbico: {f_123_p3:.4f}")

# Visualização
x_fine = np.linspace(0, 2, 100)  # Região de interesse
y_p1 = P1(x_fine)
y_p2 = P2(x_fine)
y_p3 = P3(x_fine)

plt.figure(figsize=(10, 6))
plt.plot(x[:5], y[:5], 'ro', label='Dados originais (0 a 2)')  # Até x=2
plt.plot(x_fine, y_p1, 'g--', label='Linear (grau 1)')
plt.plot(x_fine, y_p2, 'b-.', label='Quadrático (grau 2)')
plt.plot(x_fine, y_p3, 'm-', label='Cúbico (grau 3)')
plt.plot(1.23, f_123_p3, 'k*', label=f'f(1.23) cúbico = {f_123_p3:.4f}', markersize=10)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Interpolação Linear, Quadrática e Cúbica')
plt.legend()
plt.grid(True)
plt.show()

# Verificação nos pontos originais
print("\nVerificação nos pontos originais:")
for xi, yi in zip(x[:5], y[:5]):
    print(f"x = {xi}: Real = {yi}, P1 = {P1(xi):.4f}, P2 = {P2(xi):.4f}, P3 = {P3(xi):.4f}")
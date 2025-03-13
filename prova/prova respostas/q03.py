import random
import numpy as np
import matplotlib.pyplot as plt

# Definir o número de amostras
num_samples = 100

# Gerar X aleatórios no intervalo [0, 20]
X = [random.uniform(0, 20) for _ in range(num_samples)]

# Gerar a aleatória no intervalo [-5, 5]
a = [random.uniform(-5, 5) for _ in range(num_samples)]

# Calcular Y de acordo com a fórmula Y = e^X + a
Y = [np.exp(x) + ai for x, ai in zip(X, a)]


# Plotar os resultados
plt.scatter(X, Y, color='blue', label='Y = e^X + a')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico de Y = e^X + a')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Definição da função que vamos integrar
def func_a(x):
    return np.exp(x)

def func_b(x):
    return np.sqrt(x)

def func_c(x):
    return 1 / np.sqrt(x)

# Regra dos Trapézios
def trapezoidal_rule(f, a, b, n):
    # Calcula o passo (h)
    h = (b - a) / n
    # Calcula a soma inicial (meio das extremidades)
    s = (f(a) + f(b)) / 2
    # Soma os valores intermediários
    for i in range(1, n):
        s += f(a + i * h)
    # Multiplica o resultado pelo tamanho do passo
    return s * h

# Regra de Simpson
def simpson_rule(f, a, b, n):
    # Verifica se n é par (Simpson só funciona para n par)
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    # Calcula a soma inicial
    s = f(a) + f(b)
    # Soma os valores das posições ímpares multiplicados por 4
    for i in range(1, n, 2):
        s += 4 * f(a + i * h)
    # Soma os valores das posições pares multiplicados por 2
    for i in range(2, n, 2):
        s += 2 * f(a + i * h)
    # Multiplica o resultado pelo tamanho do passo dividido por 3
    return s * h / 3

# Definição dos intervalos e número de divisões
intervalos = [
    (1, 2),  # para a)
    (1, 4),  # para b)
    (2, 14)  # para c)
]

# Número de divisões (4 e 6)
divisoes = [4, 6]

# Cálculo das integrais para cada função
resultados_trapezio = {}
resultados_simpson = {}

for i, (a, b) in enumerate(intervalos):
    func = [func_a, func_b, func_c][i]
    resultados_trapezio[i] = []
    resultados_simpson[i] = []
    for n in divisoes:
        # Regra dos Trapézios
        trap = trapezoidal_rule(func, a, b, n)
        resultados_trapezio[i].append(trap)
        
        # Regra de Simpson
        simp = simpson_rule(func, a, b, n)
        resultados_simpson[i].append(simp)

# Mostrar os resultados
for i, (a, b) in enumerate(intervalos):
    func_name = ["e^x", "√x", "1/√x"][i]
    print(f"Função: {func_name}, Intervalo: [{a}, {b}]")
    for j, n in enumerate(divisoes):
        print(f"  Divisões: {n}")
        print(f"    Trapézios: {resultados_trapezio[i][j]:.6f}")
        print(f"    Simpson: {resultados_simpson[i][j]:.6f}")
    print()

# Plotando os resultados
labels = [f"f({i+1})" for i in range(3)]
x = np.array([4, 6])
"""
# Plot para a regra dos Trapézios
plt.figure(figsize=(12, 6))
for i, (a, b) in enumerate(intervalos):
    plt.plot(x, resultados_trapezio[i], label=f"Trapézios: {labels[i]}")

plt.title("Comparação entre Trapézios")
plt.xlabel("Número de divisões")
plt.ylabel("Resultado")
plt.legend()
plt.grid(True)

# Plot para a regra de Simpson
plt.figure(figsize=(12, 6))
for i, (a, b) in enumerate(intervalos):
    plt.plot(x, resultados_simpson[i], label=f"Simpson: {labels[i]}")

plt.title("Comparação entre Simpson")
plt.xlabel("Número de divisões")
plt.ylabel("Resultado")
plt.legend()
plt.grid(True)

plt.show()
"""
import numpy as np
import matplotlib.pyplot as plt

# Gerar os dados
np.random.seed(42)
x = np.linspace(0, 20, 100)
a = np.random.uniform(-5, 5, size=x.size)
y = np.exp(x) + a

# Normalizar os dados para evitar overflow
y_normalizado = y - np.exp(20)
def modelo(x, b, c):
    return b * (np.exp(x) - np.exp(20)) + c

# Função de custo
def custo(parametros, x, y):
    b, c = parametros
    y_pred = modelo(x, b, c)
    return np.sum((y - y_pred) ** 2)

# Gradiente descendente
def gradiente_descendente(x, y, b_inicial=1.0, c_inicial=0.0, taxa_aprendizado=1e-8, iteracoes=10000):
    b, c = b_inicial, c_inicial
    n = len(x)
    
    for _ in range(iteracoes):
        y_pred = modelo(x, b, c)
        erro = y - y_pred
        grad_b = -2 * np.sum(erro * (np.exp(x) - np.exp(20))) / n
        grad_c = -2 * np.sum(erro) / n
        b -= taxa_aprendizado * grad_b
        c -= taxa_aprendizado * grad_c
    
    return b, c

# Ajustar os parâmetros
b_ajustado, c_ajustado = gradiente_descendente(x, y_normalizado)
print(f"Parâmetros ajustados: b = {b_ajustado:.4f}, c = {c_ajustado:.4f}")

# Gerar a curva ajustada (desnormalizando para o gráfico)
y_ajustado = modelo(x, b_ajustado, c_ajustado) + np.exp(20)

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Dados gerados", s=20, color="blue", alpha=0.6)  # Pontos em dispersão
plt.plot(x, y_ajustado, color="red", label=f"Ajuste: y = {b_ajustado:.2f}(e^x - e^20) + {c_ajustado:.2f} + e^20", linewidth=2)  # Linha ajustada
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste Não Linear: Dados (dispersão) e Curva Ajustada (linha)")
plt.grid(True)
plt.show()
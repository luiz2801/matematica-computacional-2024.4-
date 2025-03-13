import numpy as np
import matplotlib.pyplot as plt

# Gerando dados aleatórios
np.random.seed(42)  # Para reprodutibilidade
tam_dados = 20  # Número de pontos
x_dados = np.linspace(0, 20, tam_dados)  # Valores de X entre 0 e 20
aleatorio = np.random.uniform(-5, 5, size=tam_dados)  # Ruído aleatório
y_dados = np.exp(x_dados) + aleatorio  # Aplicando a função com ruído

# Transformação logarítmica para ajuste linear
log_y = np.log(y_dados)

# Ajuste linear usando o método dos mínimos quadrados
B, A_log = np.polyfit(x_dados, log_y, 1)  # Regressão linear

# Convertendo de volta os coeficientes
A = np.exp(A_log)

# Função ajustada
x_fit = np.linspace(0, 20, 1000)
y_fit = A * np.exp(B * x_fit)

# Plotando os dados reais e a curva ajustada
plt.scatter(x_dados, y_dados, color='red', label='Dados experimentais')
plt.plot(x_fit, y_fit, color='blue', label='Curva Ajustada')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ajuste Não Linear: y = a * exp(bx)')
plt.legend()
plt.grid()
plt.show()

# Exibir os coeficientes ajustados
print(f"Coeficiente a ajustado: {A}")
print(f"Coeficiente b ajustado: {B}")
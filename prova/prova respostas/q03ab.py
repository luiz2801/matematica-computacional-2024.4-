import numpy as np  # Importa a biblioteca NumPy para operações numéricas
import matplotlib.pyplot as plt  # Importa Matplotlib para visualização dos dados

def ajuste_linear_manual(x, y):
    """
    Realiza o ajuste linear pelo método dos mínimos quadrados.
    
    Parâmetros:
    x - Array NumPy contendo os valores de x.
    y - Array NumPy contendo os valores de y.

    Retorna:
    a - Coeficiente angular (inclinação da reta).
    b - Coeficiente linear (intercepto da reta com o eixo y).
    """
    n = len(x)  # Número de pontos
    sum_x = np.sum(x)  # Soma dos valores de x
    sum_y = np.sum(y)  # Soma dos valores de y
    sum_xy = np.sum(x * y)  # Soma dos produtos xi * yi
    sum_x2 = np.sum(x ** 2)  # Soma dos quadrados de x

    # Cálculo dos coeficientes da equação da reta y = ax + b
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - (sum_x ** 2))
    b = (sum_y - a * sum_x) / n

    # Plot dos dados e da reta ajustada
    plt.scatter(x, y, label="Dados")  # Plota os pontos reais
    plt.plot(x, a*x + b, color='red', label=f"Ajuste Linear: y = {a:.2f}x + {b:.2f}")  # Plota a reta ajustada
    plt.legend()  # Adiciona legenda
    plt.show()  # Exibe o gráfico

    return a, b  # Retorna os coeficientes da equação linear


def ajuste_exponencial_manual(x, y):
    """
    Realiza o ajuste exponencial pelo método dos mínimos quadrados com transformação logarítmica.
    O modelo é da forma y = a * e^(b * x).

    Parâmetros:
    x - Array NumPy contendo os valores de x.
    y - Array NumPy contendo os valores de y.

    Retorna:
    a - Coeficiente base da exponencial.
    b - Expoente do modelo exponencial.
    """
    ln_y = np.log(y)  # Aplica logaritmo natural nos valores de y para linearizar o modelo
    n = len(x)  # Número de pontos
    sum_x = np.sum(x)  # Soma dos valores de x
    sum_ln_y = np.sum(ln_y)  # Soma dos valores transformados ln(y)
    sum_x_ln_y = np.sum(x * ln_y)  # Soma dos produtos xi * ln(yi)
    sum_x2 = np.sum(x ** 2)  # Soma dos quadrados de x

    # Cálculo dos coeficientes da equação linearizada ln(y) = ln(a) + bx
    b = (n * sum_x_ln_y - sum_x * sum_ln_y) / (n * sum_x2 - (sum_x ** 2))
    ln_a = (sum_ln_y - b * sum_x) / n
    a = np.exp(ln_a)  # Retorna ao domínio original aplicando exponencial

    # Plot dos dados e da curva ajustada
    plt.scatter(x, y, label="Dados")  # Plota os pontos reais
    plt.plot(x, a * np.exp(b * x), color='green', label=f"Ajuste Exponencial: y = {a:.2f}e^({b:.2f}x)")  # Plota a curva ajustada
    plt.legend()  # Adiciona legenda
    plt.show()  # Exibe o gráfico

    return a, b  # Retorna os coeficientes da equação exponencial




def ajuste_exponencial_e_x(x, y):
    """
    Ajuste exponencial no formato y = a * e^x pelo método dos mínimos quadrados.

    Parâmetros:
    x - Array NumPy contendo os valores de x.
    y - Array NumPy contendo os valores de y.

    Retorna:
    a - Coeficiente do modelo y = a * e^x.
    """
    ln_y = np.log(y)  # Aplicamos logaritmo natural nos valores de y
    n = len(x)

    # Cálculo do coeficiente ln(a)
    ln_a = (np.sum(ln_y) - np.sum(x)) / n
    a = np.exp(ln_a)  # Recuperamos 'a' aplicando exponencial

    # Plot dos dados e da curva ajustada
    plt.scatter(x, y, label="Dados")
    plt.plot(x, a * np.exp(x), color='green', label=f"Ajuste Exponencial: y = {a:.2f}e^x")
    plt.legend()
    plt.show()

    return a



# Exemplo de uso
x = np.array([1, 2, 3, 4, 5])  # Valores de x

# Dados para ajuste linear
y_linear = np.array([2, 3, 5, 7, 11])  # Exemplo de conjunto de dados que segue tendência linear

# Dados para ajuste exponencial
y_exp = np.array([2, 4.5, 9.1, 19, 39])  # Exemplo de conjunto de dados que segue crescimento exponencial

# Exemplo de uso
x = np.array([1, 2, 3, 4, 5])
y_exp = np.array([2, 5.4, 14.7, 40.1, 109])  # Exemplo de dados que seguem o formato y = a * e^x

# Chamada da função para ajuste
ajuste_exponencial_e_x(x, y_exp)
ajuste_linear_manual(x, y_linear)  # Ajuste linear e exibição do gráfico
ajuste_exponencial_manual(x, y_exp)  # Ajuste exponencial e exibição do gráfico

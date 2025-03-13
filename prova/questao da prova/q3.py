# Importando as bibliotecas necessárias
import numpy as np  # Para trabalhar com arrays e operações matemáticas
import matplotlib.pyplot as plt  # Para criar gráficos

def generate_data():
    """
    Gera dados simulados para ajustar os modelos. 
    A função gera um vetor de valores x e um vetor y, sendo y = exp(x) + a,
    onde 'a' é um erro aleatório.
    """
    np.random.seed(42)  # Define a semente para garantir que a geração de números aleatórios seja reprodutível
    x = np.linspace(0, 20, 50)  # Gera 50 pontos no intervalo de 0 a 20
    a = np.random.uniform(-5, 5, size=len(x))  # Gera erros aleatórios uniformemente distribuídos entre -5 e 5
    y = np.exp(x) + a  # Gera os valores de y baseados na função exponencial exp(x) com erro adicionado
    return x, y  # Retorna os valores de x e y

def linear_fit(x, y):
    """
    Realiza o ajuste linear utilizando o método dos mínimos quadrados.
    O modelo de ajuste é uma reta, ou seja, y = m*x + b.
    """
    A = np.vstack([x, np.ones(len(x))]).T  # Cria a matriz A com x e um vetor de 1s para o termo constante
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]  # Resolve o sistema de equações usando o método dos mínimos quadrados
    return coeffs[0] * x + coeffs[1]  # Retorna a reta ajustada com os coeficientes encontrados

def nonlinear_fit(x, y):
    """
    Realiza o ajuste não linear com base na função exponencial.
    Ajusta os dados ao modelo y = exp(x) + constante.
    """
    return np.exp(x) + np.mean(y - np.exp(x))  # Ajuste não linear: exp(x) mais a média dos resíduos

def plot_fits():
    """
    Gera os dados, aplica os ajustes linear e não linear, e plota os resultados.
    """
    x, y = generate_data()  # Gera os dados simulados
    
    # Ajuste Linear
    y_linear_fit = linear_fit(x, y)  # Realiza o ajuste linear
    
    # Ajuste Não Linear
    y_nonlinear_fit = nonlinear_fit(x, y)  # Realiza o ajuste não linear
    
    # Plotando os resultados
    plt.figure(figsize=(10, 6))  # Define o tamanho da figura
    plt.scatter(x, y, label='Dados Originais', color='red')  # Plota os pontos originais (dados simulados)
    plt.plot(x, y_linear_fit, label='Ajuste Linear', linestyle='--')  # Plota o ajuste linear (reta)
    plt.plot(x, y_nonlinear_fit, label='Ajuste Não Linear', linestyle='-.')  # Plota o ajuste não linear (exponencial)
    
    # Rótulos e título
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()  # Exibe a legenda
    plt.title("Ajuste Linear vs Não Linear")  # Título do gráfico
    plt.grid()  # Adiciona uma grade ao gráfico
    plt.show()  # Exibe o gráfico

# Chama a função para gerar o gráfico e aplicar os ajustes
plot_fits()

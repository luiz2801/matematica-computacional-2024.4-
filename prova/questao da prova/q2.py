# Importando as bibliotecas necessárias
import numpy as np  # Para trabalhar com arrays e operações matemáticas
import matplotlib.pyplot as plt  # Para criar gráficos

def interpolacao_polinomial(x, y, x_interp):
    """
    Calcula o polinômio interpolador de Lagrange e retorna o valor interpolado em x_interp.
    O polinômio interpolador de Lagrange é utilizado para interpolar valores em uma função
    dada uma série de pontos (x, y).
    """
    n = len(x)  # Número de pontos conhecidos
    p = np.zeros(n)  # Vetor para armazenar os termos do polinômio
    
    # Loop para calcular o valor do polinômio interpolador de Lagrange
    for i in range(n):
        termo = y[i]  # O valor de y[i] é o valor inicial do termo do polinômio
        for j in range(n):
            # Para cada par de pontos (x[i], y[i]) e (x[j], y[j]) diferentes,
            # calcula o termo do polinômio multiplicado por (x_interp - x[j]) / (x[i] - x[j]).
            if i != j:
                termo *= (x_interp - x[j]) / (x[i] - x[j])
        p[i] = termo  # Armazena o termo calculado no vetor p
    
    # Retorna a soma de todos os termos, que é o valor do polinômio interpolador em x_interp
    return np.sum(p)

def plot_interpolation():
    """
    Plota os pontos conhecidos e o polinômio interpolador.
    A função cria um gráfico onde são mostrados os pontos dados e o polinômio
    interpolador de Lagrange, além de calcular um valor interpolado para um ponto específico.
    """
    # Definindo os pontos conhecidos (x, y)
    x = np.array([0, 0.5, 1, 1.5, 2, 25])  # Valores de x conhecidos
    y = np.array([-2.78, -2.241, -1.65, -0.594, 1.34, 4.564])  # Valores de y correspondentes aos x
    
    # Gerando uma série de valores de x para avaliação do polinômio interpolador (100 pontos entre 0 e 2)
    x_vals_fine = np.linspace(0, 2, 100)
    
    # Calculando o valor interpolado para cada valor de x usando o polinômio de Lagrange
    y_interp = [interpolacao_polinomial(x, y, xi) for xi in x_vals_fine]
    
    # Criando o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals_fine, y_interp, label='Polinômio Interpolador', linestyle='--')  # Polinômio interpolador
    plt.scatter(x, y, color='red', label='Pontos Dados')  # Marcando os pontos conhecidos no gráfico
    
    # Adicionando rótulos e título ao gráfico
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()  # Adiciona uma legenda ao gráfico
    plt.title("Interpolação Polinomial")  # Título do gráfico
    plt.grid()  # Adiciona uma grade ao gráfico
    plt.show()  # Exibe o gráfico
    
    # Cálculo do valor interpolado para x = 1.23
    f_1_23 = interpolacao_polinomial(x, y, 1.23)  # Calcula o valor interpolado para x = 1.23
    print(f"O valor interpolado de f(1.23) é aproximadamente {f_1_23:.4f}")  # Exibe o valor interpolado

# Chamando a função para gerar o gráfico e calcular o valor interpolado
plot_interpolation()

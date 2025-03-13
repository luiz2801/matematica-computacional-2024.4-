# Importando as bibliotecas necessárias
import numpy as np  # Para trabalhar com arrays e operações matemáticas
import matplotlib.pyplot as plt  # Para plotar gráficos

def runge_function(x):
    """
    Define a função de Runge: f(x) = 1 / (1 + 25 * x^2)
    Essa função é conhecida por causar fenômenos como o sobreajuste em 
    interpolação polinomial.
    """
    return 1 / (1 + 25 * x**2)

f = lambda x : 1/(1+25*x**2)


def quadratic_spline_interpolation(valores_x, valores_y, x_interp):
    """
    Função para calcular a interpolação spline quadrática manualmente.
    A interpolação spline quadrática usa polinômios de grau 2 para aproximar a função
    entre os pontos.
    """
    n = len(valores_x) - 1  # Número de intervalos, pois o número de polinômios é um a menos que o número de pontos
    a = valores_y  # Os valores de y nos pontos dados
    b = np.zeros(n)  # Coeficientes b a serem calculados
    c = np.zeros(n + 1)  # Coeficientes c a serem calculados
    h = [valores_x[i + 1] - valores_x[i] for i in range(n)]  # Distâncias entre os pontos x consecutivos
    
    # Matrizes para resolver o sistema linear
    A = np.zeros((n + 1, n + 1))  # Matriz A para o sistema linear
    B = np.zeros(n + 1)  # Vetor B para o sistema linear
    
    A[0][0] = 1  # Condição de contorno natural no primeiro ponto
    A[n][n] = 1  # Condição de contorno natural no último ponto
    
    # Preenchendo as linhas do sistema para os pontos internos
    for i in range(1, n):
        A[i][i - 1] = h[i - 1]  # Elemento abaixo da diagonal principal
        A[i][i] = 2 * (h[i - 1] + h[i])  # Elemento na diagonal principal
        A[i][i + 1] = h[i]  # Elemento acima da diagonal principal
        # A equação que relaciona os coeficientes c dos splines quadráticos
        B[i] = (3 / h[i]) * (a[i + 1] - a[i]) - (3 / h[i - 1]) * (a[i] - a[i - 1])
    
    # Resolvendo o sistema linear para os coeficientes c
    c = np.linalg.solve(A, B)
    
    # Calculando os coeficientes b e d para os polinômios quadráticos
    d = np.zeros(n)  # Coeficientes d a serem calculados
    for i in range(n):
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3  # Coeficiente b
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])  # Coeficiente d
    
    # Calculando o valor interpolado para um ponto específico
    for i in range(n):
        # Verificando se o ponto de interpolação está no intervalo do spline
        if valores_x[i] <= x_interp <= valores_x[i + 1]:
            dx = x_interp - valores_x[i]  # Distância do ponto de interpolação ao ponto x_i
            # Fórmula do polinômio spline quadrático para o intervalo [x_i, x_{i+1}]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    
    # Se o valor de x_interp não está no intervalo fornecido
    raise ValueError("x_interp está fora do intervalo fornecido.")

def plot_interpolation():
    """
    Função para plotar a função de Runge e as interpolações polinomiais com diferentes números de pontos.
    A função de Runge é plotada junto com as interpolações spline quadráticas para ver como a interpolação melhora
    com o aumento do número de pontos.
    """
    num_points = [6, 10, 14]  # Diferentes números de pontos amostrados para a interpolação
    x_vals_fine = np.linspace(-2, 2, 500)  # Valores de x para plotar a função de Runge com resolução fina
    y_true = f(x_vals_fine)  # Valores de y correspondentes à função de Runge
    
    # Inicializando a figura para o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals_fine, y_true, 'k-', label='Função de Runge (Real)')  # Função de Runge (linha preta)
    
    # Loop para diferentes números de pontos
    for n in num_points:
        x_sample = np.linspace(-2, 2, n)  # Pontos amostrados no intervalo [-2, 2]
        y_sample = f(x_sample)  # Valores de y correspondentes aos pontos amostrados
        
        # Calculando a interpolação para cada ponto amostrado
        y_interp = [quadratic_spline_interpolation(x_sample, y_sample, x) for x in x_vals_fine]
        
        # Plotando a interpolação spline para o número de pontos atual
        plt.plot(x_vals_fine, y_interp, '--', label=f'Spline Quadrática ({n} pontos)')
        plt.scatter(x_sample, y_sample, marker='o', label=f'Amostras ({n} pontos)')  # Plotando os pontos amostrados
    
    # Definindo rótulos e título para o gráfico
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()  # Adicionando a legenda
    plt.title("Interpolação Spline Quadrática da Função de Runge")  # Título do gráfico
    plt.grid()  # Adicionando grade ao gráfico
    plt.show()  # Exibindo o gráfico

# Chamando a função para gerar o gráfico
plot_interpolation()

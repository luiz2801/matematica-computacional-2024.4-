import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns

# Função para calcular a tabela de diferenças divididas de Newton
def divided_differences(x_points, y_points):
    n = len(x_points)
    diff_table = np.zeros((n, n))  # Criação da tabela de diferenças divididas
    diff_table[:, 0] = y_points  # A primeira coluna é preenchida com os valores de f(x)

    # Preenchendo a tabela de diferenças divididas
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (x_points[i + j] - x_points[i])

    return diff_table

# Função para calcular o polinômio de Newton e avaliar os pontos fornecidos
def newton_interpolation(x_points, y_points, x_vals):
    n = len(x_points)
    diff_table = divided_differences(x_points, y_points)  # Calcula a tabela de diferenças divididas
    n_vals = len(x_vals)
    y_vals = np.zeros(n_vals)

    for i in range(n_vals):
        result = y_points[0]  # Começa com o primeiro valor de y
        product_term = 1
        for j in range(1, n):
            product_term *= (x_vals[i] - x_points[j - 1])  # Multiplicação dos termos (x - x_i)
            result += diff_table[0, j] * product_term  # Soma os termos do polinômio
        y_vals[i] = result  # Armazena o resultado

    return y_vals

# Função para calcular a derivada do polinômio de Newton simbolicamente
def derivative_newton(x_points, y_points, x_vals):
    n = len(x_points)
    diff_table = divided_differences(x_points, y_points)
    x = sp.symbols('x')
    poly = 0

    # Construindo o polinômio de Newton simbolicamente
    for i in range(n):
        term = diff_table[0, i]
        for j in range(i):
            term *= (x - x_points[j])
        poly += term

    # Derivando o polinômio
    derivative = sp.diff(poly, x)

    # Avaliando a derivada nos pontos fornecidos
    derivative_vals = [derivative.subs(x, val) for val in x_vals]
    return derivative_vals

# Função para calcular a integral do polinômio de Newton
def integral_newton(x_points, y_points, a, b):
    n = len(x_points)
    diff_table = divided_differences(x_points, y_points)
    x = sp.symbols('x')
    poly = 0

    # Construindo o polinômio de Newton simbolicamente
    for i in range(n):
        term = diff_table[0, i]
        for j in range(i):
            term *= (x - x_points[j])
        poly += term

    # Integrando o polinômio
    integral = sp.integrate(poly, (x, a, b))
    return integral

# Função para encontrar o menor grau do polinômio que fornece boa precisão
def find_best_degree(x_points, y_points, x_target):
    n = len(x_points)
    best_degree = None
    best_approx = None
    min_error = float('inf')

    for degree in range(1, n):
        approx = newton_interpolation(x_points[:degree+1], y_points[:degree+1], [x_target])[0]
        error = abs(approx - newton_interpolation(x_points, y_points, [x_target])[0])

        if error < min_error:
            min_error = error
            best_degree = degree
            best_approx = approx

        if error < 1e-3:  # Definimos um critério de erro aceitável
            break

    return best_degree, best_approx


def plot_divided_differences(x_points, y_points):
    diff_table = divided_differences(x_points, y_points)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(diff_table, annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5)
    plt.title("Tabela de Diferenças Divididas de Newton")
    plt.xlabel("Ordem da Diferença")
    plt.ylabel("Índice dos Pontos")
    plt.show()

# Função principal que executa a interpolação e outras operações
def main():
    try:
        # Definição dos pontos fornecidos
        x_points = np.array([0, 0.5, 1, 1.5, 2, 2.5])
        y_points = np.array([-2.78, -2.241, -1.65, -0.594, 1.34, 4.564])
        plot_divided_differences(x_points, y_points)

        # Determinar o menor grau do polinômio que fornece boa precisão
        x_target = 1.23
        best_degree, best_approx = find_best_degree(x_points, y_points, x_target)

        print(f"Menor grau do polinômio com boa precisão: {best_degree}")
        print(f"Valor interpolado de f(1.23) usando esse grau: {best_approx}")

        # Gerar a interpolação para plotagem
        x_vals = np.linspace(min(x_points), max(x_points), 100)
        y_vals = newton_interpolation(x_points, y_points, x_vals)

        # Plotar a interpolação
        plt.plot(x_vals, y_vals, label="Polinômio Interpolador")
        plt.scatter(x_points, y_points, color='red', label="Pontos dados")
        plt.scatter([x_target], [best_approx], color='green', label="f(1.23)")
        plt.legend()
        plt.title("Interpolação Polinomial de Newton")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.show()

        # Solicitar ao usuário qual operação deseja realizar
        print("\nEscolha a operação que você deseja realizar com a interpolação:")
        print("1. Derivada do polinômio interpolador")
        print("2. Integral do polinômio interpolador")
        operation = int(input("Escolha uma operação (1/2): "))

        if operation == 1:
            x_val = float(input("Informe o valor de x para calcular a derivada: "))
            derivative_vals = derivative_newton(x_points, y_points, [x_val])
            print(f"A derivada do polinômio em x = {x_val} é: {derivative_vals[0]}")

        elif operation == 2:
            a = float(input("Informe o limite inferior da integral: "))
            b = float(input("Informe o limite superior da integral: "))
            integral_value = integral_newton(x_points, y_points, a, b)
            print(f"A integral do polinômio interpolador no intervalo [{a}, {b}] é: {integral_value}")

        else:
            print("Operação inválida. Saindo...")

    except Exception as e:
        print(f"Ocorreu um erro durante o processo: {e}")



if __name__ == "__main__":
    main()

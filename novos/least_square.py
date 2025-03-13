import numpy as np
import matplotlib.pyplot as plt

# Função para obter os pontos do usuário
def get_user_points():
    print("Enter your points (x,y) one per line")
    print("Enter an empty line when finished")
    
    x_points = []  # Lista para armazenar os valores de x
    y_points = []  # Lista para armazenar os valores de y
    
    while True:
        point = input("Enter x,y coordinates (e.g., '2,3'): ")
        if point.strip() == "":
            break  # Sai do loop se a entrada estiver vazia
        try:
            x, y = map(float, point.split(','))  # Converte a entrada para dois números flutuantes
            if x <= 0 and len(x_points) > 0:  # Verifica se x é positivo para ajuste logarítmico
                print("Warning: x must be positive for logarithmic fit")
                continue
            x_points.append(x)  # Adiciona o valor de x à lista
            y_points.append(y)  # Adiciona o valor de y à lista
        except ValueError:
            print("Invalid format. Please use 'x,y' format with numbers")
            continue  # Se a entrada não for válida, solicita novamente
    
    if len(x_points) < 2:
        raise ValueError("Need at least 2 points for fitting")  # Verifica se há pelo menos 2 pontos
    
    return np.array(x_points), np.array(y_points)  # Retorna os arrays de x e y como numpy

# Função para ajustar a função de acordo com o tipo escolhido
def fit_function(x, y, fit_type):
    if fit_type == 'linear':
        # Ajuste linear: y = mx + b
        A = np.vstack([x, np.ones(len(x))]).T  # Cria a matriz de design para regressão linear
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]  # Resolve para os coeficientes (m, b)
        return coeffs, lambda x: coeffs[0]*x + coeffs[1], 'y = {:.4f}x + {:.4f}'
    
    elif fit_type == 'quadratic':
        # Ajuste quadrático: y = ax² + bx + c
        A = np.vstack([x**2, x, np.ones(len(x))]).T  # Cria a matriz de design para regressão quadrática
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]  # Resolve para os coeficientes (a, b, c)
        return coeffs, lambda x: coeffs[0]*x**2 + coeffs[1]*x + coeffs[2], 'y = {:.4f}x² + {:.4f}x + {:.4f}'
    
    elif fit_type == 'exponential':
        # Ajuste exponencial: y = ae^(bx)
        y_shift = 0
        if min(y) <= 0:
            y_shift = abs(min(y)) + 1  # Desloca os valores de y se algum for menor ou igual a zero
        y_transformed = np.log(y + y_shift)  # Transforma os valores de y com logaritmo natural
        A = np.vstack([x, np.ones(len(x))]).T  # Matriz de design para ajuste exponencial
        coeffs = np.linalg.lstsq(A, y_transformed, rcond=None)[0]  # Resolve para a e b
        a = np.exp(coeffs[1])  # Exponencia para obter o coeficiente a
        b = coeffs[0]  # Coeficiente b
        return [a, b], lambda x: a * np.exp(b * x) - y_shift, 'y = {:.4f}e^({:.4f}x)'
    
    elif fit_type == 'logarithmic':
        # Ajuste logarítmico: y = a + b*ln(x)
        A = np.vstack([np.log(x), np.ones(len(x))]).T  # Cria a matriz de design para regressão logarítmica
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]  # Resolve para os coeficientes (a, b)
        return coeffs, lambda x: coeffs[0]*np.log(x) + coeffs[1], 'y = {:.4f}ln(x) + {:.4f}'

# Função para calcular o valor de y para um x passado
def predict_y_value(fit_func):
    try:
        x_input = float(input("Enter the value of x for prediction: "))  # Pede ao usuário para inserir um valor de x
        y_pred = fit_func(x_input)  # Calcula o valor de y com a função de ajuste
        print(f"Predicted value of y for x = {x_input}: {y_pred:.4f}")
    except ValueError:
        print("Invalid input. Please enter a valid number.")  # Se o input não for válido

# Obter os pontos do usuário, com tratamento de erro
try:
    x, y = get_user_points()
except ValueError as e:
    print(f"Error: {e}")
    print("Using sample data instead...")
    np.random.seed(42)
    x = np.linspace(1, 10, 20)  # Gera valores de x entre 1 e 10
    y = 2 * np.log(x) + 1 + np.random.normal(0, 0.5, 20)  # Gera valores de y com um pouco de ruído

# Exibir tipos de ajuste disponíveis
print("\nAvailable fit types:")
print("1: Linear (y = mx + b)")
print("2: Quadratic (y = ax² + bx + c)")
print("3: Exponential (y = ae^(bx))")
print("4: Logarithmic (y = a + bln(x))")
fit_choice = input("Choose fit type (1-4): ")

# Mapear a escolha do usuário para o tipo de ajuste correspondente
fit_types = {'1': 'linear', '2': 'quadratic', '3': 'exponential', '4': 'logarithmic'}
fit_type = fit_types.get(fit_choice, 'linear')  # Ajuste padrão para linear se a escolha for inválida

# Verificar se o ajuste logarítmico foi escolhido e garantir que todos os valores de x sejam positivos
if fit_type == 'logarithmic' and min(x) <= 0:
    print("Error: Logarithmic fit requires all x values to be positive")
    print("Switching to linear fit")
    fit_type = 'linear'

# Realizar o ajuste com o tipo selecionado
coeffs, fit_func, eq_format = fit_function(x, y, fit_type)

# Gerar os dados da curva ajustada para o gráfico
x_fit = np.linspace(max(min(x), 0.001 if fit_type == 'logarithmic' else min(x)), max(x), 100)
y_fit = fit_func(x_fit)

# Plotar os pontos de dados originais e a curva ajustada
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_fit, y_fit, color='red', label=f'Fit: {eq_format.format(*coeffs)}')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Least Squares {fit_type.capitalize()} Fit')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir a equação do ajuste
print(f"\nResults ({fit_type} fit):")
print(f"Fitted function: {eq_format.format(*coeffs)}")

# Calcular e imprimir o R-quadrado (qualidade do ajuste)
y_pred = fit_func(x)
residuals = y - y_pred
ss_res = np.sum(residuals**2)  # Soma dos resíduos quadrados
ss_tot = np.sum((y - np.mean(y))**2)  # Soma total dos quadrados
r_squared = 1 - (ss_res / ss_tot)  # Fórmula do R-quadrado
print(f"R-squared: {r_squared:.4f}")

# Perguntar se o usuário quer calcular o valor de y para um novo x
calculate_for_x = input("Do you want to calculate the value of y for a specific x? (yes/no): ")
if calculate_for_x.lower() == 'yes':
    predict_y_value(fit_func)  # Chama a função para prever o valor de y

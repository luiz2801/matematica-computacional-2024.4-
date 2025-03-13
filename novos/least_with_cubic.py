import numpy as np
import matplotlib.pyplot as plt

# Função para obter pontos do usuário
def get_user_points():
    print("Enter your points (x,y) one per line")
    print("Enter an empty line when finished")
    
    x_points = []  # Lista para armazenar os valores de x
    y_points = []  # Lista para armazenar os valores de y
    
    while True:
        point = input("Enter x,y coordinates (e.g., '2,3'): ")
        if point.strip() == "":  # Quando o usuário pressiona Enter sem digitar nada, sai do loop
            break
        try:
            # Divide a entrada em x e y, e converte para float
            x, y = map(float, point.split(','))
            
            # Verifica se o valor de x é válido para um ajuste logarítmico
            if x <= 0 and len(x_points) > 0:  # x deve ser positivo para o ajuste logarítmico
                print("Warning: x must be positive for logarithmic fit")
                continue  # Ignora este ponto e solicita outro
            
            x_points.append(x)  # Adiciona o valor de x à lista
            y_points.append(y)  # Adiciona o valor de y à lista
        except ValueError:
            print("Invalid format. Please use 'x,y' format with numbers")
            continue  # Se a entrada não for válida, solicita novamente
    
    if len(x_points) < 2:  # Verifica se há pelo menos 2 pontos
        raise ValueError("Need at least 2 points for fitting")
    
    return np.array(x_points), np.array(y_points)  # Retorna os pontos como arrays do numpy

# Função para realizar o ajuste de acordo com o tipo selecionado
def fit_function(x, y, fit_type):
    if fit_type == 'linear':
        # Ajuste linear: y = mx + b
        A = np.vstack([x, np.ones(len(x))]).T  # Matriz de design para regressão linear
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]  # Coeficientes m e b
        return coeffs, lambda x: coeffs[0]*x + coeffs[1], 'y = {:.4f}x + {:.4f}'
    
    elif fit_type == 'quadratic':
        # Ajuste quadrático: y = ax² + bx + c
        A = np.vstack([x**2, x, np.ones(len(x))]).T  # Matriz de design para regressão quadrática
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]  # Coeficientes a, b e c
        return coeffs, lambda x: coeffs[0]*x**2 + coeffs[1]*x + coeffs[2], 'y = {:.4f}x² + {:.4f}x + {:.4f}'
    
    elif fit_type == 'cubic':
        # Ajuste cúbico: y = ax³ + bx² + cx + d
        A = np.vstack([x**3, x**2, x, np.ones(len(x))]).T  # Matriz de design para regressão cúbica
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]  # Coeficientes a, b, c e d
        return coeffs, lambda x: coeffs[0]*x**3 + coeffs[1]*x**2 + coeffs[2]*x + coeffs[3], 'y = {:.4f}x³ + {:.4f}x² + {:.4f}x + {:.4f}'
    
    elif fit_type == 'exponential':
        # Ajuste exponencial: y = a*e^(bx)
        y_shift = 0  # Deslocamento de y para garantir que os valores sejam positivos para logaritmos
        if min(y) <= 0:
            y_shift = abs(min(y)) + 1  # Desloca y para ser positivo
        y_transformed = np.log(y + y_shift)  # Aplica o logaritmo natural nos valores de y
        A = np.vstack([x, np.ones(len(x))]).T  # Matriz de design para ajuste exponencial
        coeffs = np.linalg.lstsq(A, y_transformed, rcond=None)[0]  # Coeficientes a e b
        a = np.exp(coeffs[1])  # Coeficiente a é a exponencial de b
        b = coeffs[0]  # Coeficiente b
        return [a, b], lambda x: a * np.exp(b * x) - y_shift, 'y = {:.4f}e^({:.4f}x)'
    
    elif fit_type == 'logarithmic':
        # Ajuste logarítmico: y = a + b*ln(x)
        A = np.vstack([np.log(x), np.ones(len(x))]).T  # Matriz de design para ajuste logarítmico
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]  # Coeficientes a e b
        return coeffs, lambda x: coeffs[0]*np.log(x) + coeffs[1], 'y = {:.4f}ln(x) + {:.4f}'

# Função para calcular o valor de y em um ponto específico de x
def calculate_y_at_x(fit_func, x_value):
    return fit_func(x_value)

# Obter pontos do usuário com tratamento de erro
try:
    x, y = get_user_points()
except ValueError as e:
    print(f"Error: {e}")
    print("Using sample data instead...")
    np.random.seed(42)
    x = np.linspace(1, 10, 20)  # Gera valores de x entre 1 e 10 (necessário para ajuste logarítmico)
    y = 2 * np.log(x) + 1 + np.random.normal(0, 0.5, 20)  # Gera valores de y com um pouco de ruído

# Apresentar as opções de ajuste
print("\nAvailable fit types:")
print("1: Linear (y = mx + b)")
print("2: Quadratic (y = ax² + bx + c)")
print("3: Cubic (y = ax³ + bx² + cx + d)")
print("4: Exponential (y = ae^(bx))")
print("5: Logarithmic (y = a + bln(x))")
fit_choice = input("Choose fit type (1-5): ")

# Mapeamento da escolha do usuário para os tipos de ajuste
fit_types = {
    '1': 'linear',
    '2': 'quadratic',
    '3': 'cubic',
    '4': 'exponential',
    '5': 'logarithmic'
}
fit_type = fit_types.get(fit_choice, 'linear')  # Padrão para linear caso a escolha seja inválida

# Verificar se o ajuste logarítmico foi selecionado e garantir que x seja positivo
if fit_type == 'logarithmic' and min(x) <= 0:
    print("Error: Logarithmic fit requires all x values to be positive")
    print("Switching to linear fit")
    fit_type = 'linear'

# Realizar o ajuste de acordo com o tipo selecionado
coeffs, fit_func, eq_format = fit_function(x, y, fit_type)

# Gerar dados da curva ajustada para o gráfico
x_fit = np.linspace(max(min(x), 0.001 if fit_type == 'logarithmic' else min(x)), max(x), 100)
y_fit = fit_func(x_fit)

# Plotar os pontos originais e a curva ajustada
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_fit, y_fit, color='red', label=f'Fit: {eq_format.format(*coeffs)}')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Least Squares {fit_type.capitalize()} Fit')
plt.legend()
plt.grid(True)
plt.show()

# Exibir os resultados
print(f"\nResults ({fit_type} fit):")
print(f"Fitted function: {eq_format.format(*coeffs)}")

# Calcular e exibir o R-quadrado (qualidade do ajuste)
y_pred = fit_func(x)
residuals = y - y_pred  # Resíduos da previsão
ss_res = np.sum(residuals**2)  # Soma dos resíduos quadrados
ss_tot = np.sum((y - np.mean(y))**2)  # Soma total dos quadrados
r_squared = 1 - (ss_res / ss_tot)  # Fórmula para calcular o R-quadrado
print(f"R-squared: {r_squared:.4f}")

# Calcular o valor de y para um ponto específico de x
x_value = float(input("\nEnter a value of x to calculate y: "))  # Solicita ao usuário um valor de x
y_value = calculate_y_at_x(fit_func, x_value)  # Calcula o valor de y usando a função ajustada
print(f"The calculated value of y for x = {x_value} is y = {y_value:.4f}")

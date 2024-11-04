def calcular_area_triangulo(base, altura):
    return (base * altura) / 2

# Exemplo de uso
base = float(input("Digite a base do triângulo: "))
altura = float(input("Digite a altura do triângulo: "))

area = calcular_area_triangulo(base, altura)
print(f"A área do triângulo é: {area}")

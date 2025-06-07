import math
import random
import emoji

print(emoji.emojize("Olá , Mundo :sunglasses:" ))
"""
def calcularMedia():
    vetor = []
    while True:
        n = float(input("Digite um número ou 0 para calcular média"))
        if n == 0:
            break
        vetor.append(n)
    if vetor:
        media = sum(vetor) / len(vetor)
        print(f"A média dos números é: {media}")
    else:
        print("Nenhum número foi digitado")

calcularMedia()        

print("{} e {}  " .format(n1- 1 , n1+ 1))
print("{}, {} e {}" .format(n1 * 2 , n1*3 , n1**2))
valorDolar = 6
valorEuro = 7

carteira = float(input("Quanto dinheiro você tem na carteira   "))

def converter(carteira, valorDolar):
    return print(f"Você pode comprar {carteira/valorDolar} dólares")

converter(carteira , valorDolar)

larg = float(input("Digite a largura"))
alt = float(input('Digite a altura'))

area = larg * alt
print(f'Sua parede tem a dimensão de {larg}x{alt} e a sua área é {area}m^2 ')
tinta= area/2

preco = float(input('Digite o preço'))
print(f'O preço com desconto é {0.95*preco} ')

temp = float(input('Digite a temp'))
print(f"A temperatura em farenheit é {1.8*temp + 32} graus")

qtdDias = int(input("Quantos dias você usou o carro"))
qtdKm = float(input("Digite a quantidade de kms rodados"))
valorAPagar = qtdDias*60 + qtdKm*0.15
print(f"Valor a pagar : {valorAPagar} ")

name = "Derick Teles"
print(len(name))
print(name.find("i"))
print(name.capitalize())
print(name.upper())
print(name.lower())
print(name.isdigit())
print(name.isalpha())
print(name.count("e"))
print(name.replace("e", "i"))
print(name*3)

# Angulos em python
an = float(input("Digite c1  "))
sen = math.sin(math.radians(an))
cos = math.cos(math.radians(an))
tan = math.tan(math.radians(an))

lista = ['Derick' , ' Alice' , ' Maria' , 'Natália' , 'Yone']
chosen = random.choice(lista)

lista = ['Derick' , ' Alice' , ' Maria' , 'Natália' , 'Yone']
random.shuffle(lista) 

nome = str(input("Digite o seu nome: ")).strip()

print(nome.upper())
print(nome.lower())
print(f"Seu nome tem {len(nome) - nome.count(' ')} letras")
print(f'Seu primeiro nome tem {nome.find(' ')} letras')

n = int(input("Digite n"))
u = n//1 % 10 
d = n//10 % 10
c = n// 100 % 10
m = n//1000 % 10

print(f"Milhar : {m}")
print(f"Centena : {c}")
print(f"Dezena : {d}")
print(f"Unidade : {u}")

cid = str(input('Digite a cidade:  ')).strip()
print(cid[:5].upper() == 'SANTO')

"""
#nome = str(input('Digite a seu nome:  ')).strip()

frase = str(input('Digite uma frase:  ')).upper.strip()
print('A letra A aparece {} vezes na frase' .format(frase.count('A')))
 






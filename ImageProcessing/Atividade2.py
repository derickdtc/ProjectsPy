
#Trabalho feito por: Derick Teles Chagas 
from skimage.transform import rescale
from skimage.io import imread
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt

def highboost_filter(image, A=2.5, sigma=0.7):    
    # Suavizar a imagem usando um filtro gaussiano. Sigma para suavização gaussiana (controle de borramento).

    smoothed = gaussian(image, sigma=sigma, mode='reflect')
    # Calcular a máscara de alta frequência
    mask = image - smoothed
    # Aplicar o filtro highboost
    highboost = image + A * mask
    # image: Imagem original em escala de cinza. A: Fator de amplificação (A > 1).

    # Normalizar os valores para 0-255
    highboost = np.clip(highboost, 0, 1)  # Como a imagem está normalizada para [0, 1]
    return highboost

# Processar todas as imagens
for i in range(1, 6):  
    filename = f'placa0{i}.png'    
    
    image = imread(filename, as_gray=True)
    # Reduzir e restaurar a escala
    scale = 1 / 4
    image_rescaled = rescale(image, scale, anti_aliasing=True)
    image_restored = rescale(image_rescaled, 1 / scale, anti_aliasing=True)    
    
    highboost_image = highboost_filter(image_restored, A=2.5, sigma=0.7)    
    
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Imagem Original')
    ax[0].axis('off')
    
    ax[1].imshow(image_rescaled, cmap='gray')
    ax[1].set_title('Reduzida')
    ax[1].axis('off')
    
    ax[2].imshow(image_restored, cmap='gray')
    ax[2].set_title('Restaurada')
    ax[2].axis('off')
    
    ax[3].imshow(highboost_image, cmap='gray')
    ax[3].set_title('Highboost')
    ax[3].axis('off')
    
    plt.tight_layout()
    plt.show()

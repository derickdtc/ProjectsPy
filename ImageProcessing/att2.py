import cv2
import numpy as np
import matplotlib.pyplot as plt

def highboost_filter(image, A=2.0):
    """
    Aplica o filtro highboost em uma imagem usando convolução eficiente.
    
    Parâmetros:
    - image: Imagem original em escala de cinza.
    - A: Fator de amplificação.
    
    Retorna:
    - Imagem com filtro highboost aplicado.
    """
    # Aplicar filtro gaussiano para suavização (filtro passa-baixa)
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    
    # Calcular a máscara de alta frequência
    mask = cv2.subtract(image, blurred)
    
    # Aplicar o filtro highboost
    highboost = cv2.addWeighted(image, 1.0, mask, A, 0)
    
    return highboost

# Processar todas as imagens
for i in range(1, 6):  # Iterar sobre placa01.png até placa05.png
    filename = f'placa0{i}.png'
    
    # Carregar a imagem
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Reduzir e restaurar a escala
    scale = 0.25
    image_rescaled = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image_restored = cv2.resize(image_rescaled, image.shape[::-1], interpolation=cv2.INTER_CUBIC)
    
    # Aplicar o filtro highboost
    highboost_image = highboost_filter(image_restored, A=2.0)
    
    # Plotar os resultados
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

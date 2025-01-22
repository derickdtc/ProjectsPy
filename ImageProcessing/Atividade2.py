from skimage.transform import rescale
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

def highboost_filter(image, A=1.5):
    """
    Aplica o filtro highboost em uma imagem.
    
    Parâmetros:
    - image: Imagem original em escala de cinza.
    - A: Fator de amplificação (A > 1).
    
    Retorna:
    - Imagem filtrada com realce de alta frequência.
    """
    # Criar uma versão suavizada da imagem (passa-baixa)
    kernel_size = 3  # Tamanho menor para preservar mais detalhes
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    smoothed = np.zeros_like(image, dtype=np.float32)
    
    # Aplicar convolução manual para suavizar a imagem
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='reflect')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            smoothed[i, j] = np.sum(region * kernel)
    
    # Calcular a máscara (alta frequência)
    mask = image - smoothed  # Não normalizar a máscara
    
    # Aplicar o filtro highboost
    highboost = image + A * mask  # Aplicar a amplificação diretamente
    
    # Normalizar para faixa 0-255
    highboost = np.clip(highboost, 0, 255)
    return highboost.astype(np.uint8)

# Processar todas as imagens
for i in range(1, 6):  # Iterar sobre placa01.png até placa05.png
    filename = f'placa0{i}.png'
    
    # Carregar a imagem
    image = imread(filename, as_gray=True)
    image = (image * 255).astype(np.uint8)  # Converter para uint8
    
    # Reduzir e restaurar a escala
    scale = 1 / 4
    image_rescaled = rescale(image, scale, anti_aliasing=True)
    image_restored = rescale(image_rescaled, 1 / scale, anti_aliasing=True)
    image_restored = (image_restored * 255).astype(np.uint8)  # Converter para uint8
    
    # Aplicar o filtro highboost
    highboost_image = highboost_filter(image_restored, A=1.5)
    
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

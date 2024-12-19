import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def local_histogram_equalization(image, window_size, clip_limit=2.0):
    """Aplica a equalização local do histograma usando CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(window_size, window_size))
    return clahe.apply(image)

# Carregar a imagem em escala de cinza
image = cv2.imread(r'C:\Users\deric\git\ProjectsPy\ImageProcessing\prova.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Erro: Não foi possível carregar a imagem. Verifique o caminho ou o formato.")
    exit()

# Aplicar a equalização local com diferentes tamanhos de janela
window_sizes = [15, 51]
processed_images = []
for size in window_sizes:
    processed_images.append(local_histogram_equalization(image, size))

# Exibir as imagens processadas e a original
plt.figure(figsize=(15, 8))

# Original
plt.subplot(1, 3, 1)
plt.title("Imagem Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Processadas
for i, (size, processed_image) in enumerate(zip(window_sizes, processed_images)):
    plt.subplot(1, 3, i + 2)
    plt.title(f"Equalização Local - Janela {size}x{size}")
    plt.imshow(processed_image, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

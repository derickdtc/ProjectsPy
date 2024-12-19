import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def local_histogram_equalization(image, window_size):
    """Aplica equalização local do histograma em uma imagem."""
    height, width = image.shape
    pad_size = window_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    equalized_image = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            local_window = padded_image[i:i + window_size, j:j + window_size]
            equalized_image[i, j] = cv2.equalizeHist(local_window)[pad_size, pad_size]
    
    return equalized_image

# Carregar imagem
image = cv2.imread(r'C:\Users\deric\git\ProjectsPy\ImageProcessing\prova.jpg', cv2.IMREAD_GRAYSCALE)

# Processar com diferentes tamanhos de janelas
window_sizes = [31, 71]  # Alterar conforme necessário
processed_images = []

for size in window_sizes:
    processed_image = local_histogram_equalization(image, size)
    processed_images.append(processed_image)
    output_filename = f'processed_window_{size}.png'
    cv2.imwrite(output_filename, processed_image)

# Mostrar e salvar resultados
for idx, img in enumerate(processed_images):
    plt.figure()
    plt.title(f"Equalização Local - Janela {window_sizes[idx]}x{window_sizes[idx]}")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(f"result_{window_sizes[idx]}.png")
plt.show()

# OCR e conversão para texto
os.system("ocrmypdf prova.jpeg output.pdf")
os.system("pdftotext output.pdf output.txt")

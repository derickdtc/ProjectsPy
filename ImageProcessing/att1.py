import numpy as np
from skimage import io, img_as_float
from skimage.exposure import equalize_hist
from skimage.filters import gaussian
import matplotlib.pyplot as plt

def local_histogram_equalization(image, window_size):
    # Obter dimensões da imagem
    height, width = image.shape
    half_window = window_size // 2
    
    # Inicializar imagem de saída
    equalized_image = np.zeros_like(image)
    
    # Percorrer a imagem com janela deslizante
    for i in range(half_window, height - half_window):
        for j in range(half_window, width - half_window):
            # Extrair janela local
            local_region = image[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
            # Equalizar histograma local
            equalized_image[i, j] = equalize_hist(local_region)[half_window, half_window]
    
    return equalized_image

# Carregar imagem
image = img_as_float(io.imread(r'C:\Users\deric\git\ProjectsPy\ImageProcessing\prova.jpg', as_gray=True))

# Suavizar imagem para reduzir ruído antes da equalização
smoothed_image = gaussian(image, sigma=1)

# Aplicar equalização local
window_sizes = [14, 28]  # Teste tamanhos de janela maiores
results = [local_histogram_equalization(smoothed_image, size) for size in window_sizes]

# Exibir resultados
plt.figure(figsize=(15, 5))
plt.subplot(1, len(results) + 1, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

for i, result in enumerate(results):
    plt.subplot(1, len(results) + 1, i + 2)
    plt.imshow(result, cmap='gray')
    plt.title(f'Equalização Local - Janela {window_sizes[i]}x{window_sizes[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

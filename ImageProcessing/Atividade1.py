import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

# Carregar a imagem
img = imread(r'C:\Users\deric\git\ProjectsPy\ImageProcessing\prova.jpeg')

# Garantir que a imagem está em escala de cinza
if img.ndim == 3:  # Imagem colorida
    img = np.mean(img, axis=2).astype(np.uint8)  # Converter para escala de cinza

# Parâmetros da janela deslizante
janela = 32  # Tamanho da janela (ajuste conforme necessário)
metade_janela = janela // 2

# Criar a imagem de saída
output = np.zeros_like(img, dtype=np.uint8)

# Dimensões da imagem
altura, largura = img.shape

# Função para calcular a transformação local
def equalizar_local(patch):
    h, _ = np.histogram(patch, bins=256, range=(0, 256))
    h = h.astype('float') / patch.size
    T = (np.cumsum(h) * 255).astype('uint8')
    return T

# Aplicar a equalização local
for i in range(altura):
    for j in range(largura):
        # Obter a janela ao redor do pixel (com bordas tratadas)
        i_min = max(i - metade_janela, 0)
        i_max = min(i + metade_janela + 1, altura)
        j_min = max(j - metade_janela, 0)
        j_max = min(j + metade_janela + 1, largura)
        
        patch = img[i_min:i_max, j_min:j_max]  # Extrair submatriz
        T = equalizar_local(patch)  # Obter transformação local
        output[i, j] = T[img[i, j]]  # Aplicar transformação

# Plotar resultados
_, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Imagem Original (Baixo Contraste)')
ax[0].axis('off')

ax[1].imshow(output, cmap='gray')
ax[1].set_title('Imagem Equalizada Localmente')
ax[1].axis('off')

plt.tight_layout()
plt.show()

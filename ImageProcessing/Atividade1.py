import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from math import floor
import os

# Carregar e converter imagem para escala de cinza
img = imread(r'C:\Users\deric\git\ProjectsPy\ImageProcessing\prova.jpeg')
img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)  # converte RGB para cinza
      
def eq_local(img, dimensao):
    img_eq = np.copy(img) 
    altura = img.shape[0]
    largura = img.shape[1]
    
    for i in range(altura): 
        for j in range(largura):  # Para cada pixel
            local = img[max(0, i - dimensao//2):min(altura, i + dimensao//2 + 1), 
                        max(0, j - dimensao//2):min(largura, j + dimensao//2 + 1)]  # Região local
            
            h, _ = np.histogram(local, bins=256, range=(0, 256))  # Calcula histograma local
            h = h.astype(float) / local.size  # Normaliza
            
            h = np.clip(h, 0, 0.005)  # Aplica clip limit
            h = h / np.sum(h)  # Normaliza novamente
            
            equalizado = (np.cumsum(h) * 255).astype('uint8')  # Calcula valores equalizados
            img_eq[i, j] = equalizado[img[i, j]]  # Aplica transformação
            
    return img_eq

# Aplicar equalização local com diferentes tamanhos de janelas
img_eq1 = eq_local(img, 80)
img_eq2 = eq_local(img, 150)

# Salvar imagens equalizadas
imsave("img_eq1.png", img_eq1)
imsave("img_eq2.png", img_eq2)

# Gerar PDFs a partir das imagens
os.system("convert img_eq1.png img_eq1.pdf")  
os.system("convert img_eq2.png img_eq2.pdf")

# Tornar PDFs pesquisáveis usando OCRmyPDF
os.system("ocrmypdf img_eq1.pdf img_eq1_ocr.pdf")
os.system("ocrmypdf img_eq2.pdf img_eq2_ocr.pdf")

# Mostrar as imagens e salvar os PDFs
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('80x80')
plt.imshow(img_eq1, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('150x150')
plt.imshow(img_eq2, cmap='gray')

plt.show()

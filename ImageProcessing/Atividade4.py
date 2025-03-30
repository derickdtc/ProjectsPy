import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import sobel
from skimage.morphology import binary_opening, binary_closing
from skimage.draw import circle_perimeter
import math

def naive_hough_circle(img_bin, radius):
    """
    Implementação ingênua da Transformada de Hough para círculos com um único raio.
    Para cada pixel de borda (valor 1) da imagem binária, varre 360° e acumula votos 
    para os possíveis centros de círculos de raio 'radius'.
    
    Parâmetros:
    -----------
    img_bin : ndarray
        Imagem binária (0 e 1) onde 1 representa pixel de borda.
    radius : int
        Raio fixo para o qual se deseja detectar círculos.
    
    Retorno:
    --------
    accumulator : ndarray
        Grade de acumuladores (mesmo tamanho que a imagem) com os votos.
        Visualmente, apresenta os "anéis" de votos provenientes dos pixels de borda.
    """
    H, W = img_bin.shape
    accumulator = np.zeros((H, W), dtype=np.uint64)
    # Precalcula os ângulos (0 a 359° em radianos)
    angulos = np.deg2rad(np.arange(0, 360, 1))
    cos_angles = np.cos(angulos)
    sin_angles = np.sin(angulos)
    
    # Encontra os pixels de borda (valor 1)
    edge_y, edge_x = np.nonzero(img_bin)
    
    # Para cada pixel de borda, vote para todos os ângulos
    for y, x in zip(edge_y, edge_x):
        # Calcula os centros candidatos:
        # a = x - r*cos(theta) e b = y - r*sin(theta)
        a_candidatos = np.rint(x - radius * cos_angles).astype(int)
        b_candidatos = np.rint(y - radius * sin_angles).astype(int)
        # Atualiza o acumulador apenas se o centro candidato estiver na imagem
        for a, b in zip(a_candidatos, b_candidatos):
            if 0 <= a < W and 0 <= b < H:
                accumulator[b, a] += 1
    return accumulator

def naive_hough_circle_peaks(accumulator, total_num_peaks=10, threshold=None, nhood_size=8):
    """
    Encontra os picos (centros prováveis) na grade de acumuladores usando supressão não-máxima.
    
    Parâmetros:
    -----------
    accumulator : ndarray
        A grade de acumuladores (resultado da CHT ingênua).
    total_num_peaks : int
        Número máximo de picos (círculos) a detectar.
    threshold : float ou None
        Valor mínimo do acumulador para ser considerado pico.
        Se None, usa 50% do valor máximo do acumulador.
    nhood_size : int
        Tamanho da vizinhança para supressão dos picos próximos.
    
    Retorno:
    --------
    acc_vals : list
        Valores do acumulador nos picos selecionados.
    centers_y : list
        Coordenadas Y dos centros detectados.
    centers_x : list
        Coordenadas X dos centros detectados.
    """
    H, W = accumulator.shape
    if threshold is None:
        threshold = 0.5 * accumulator.max()
    
    # Lista de candidatos (valor, y, x)
    candidatos = []
    for y in range(H):
        for x in range(W):
            if accumulator[y, x] >= threshold:
                candidatos.append((accumulator[y, x], y, x))
                
    # Ordena os candidatos em ordem decrescente (maior valor primeiro)
    candidatos.sort(key=lambda c: c[0], reverse=True)
    
    selecionados = []
    for val, y, x in candidatos:
        if len(selecionados) >= total_num_peaks:
            break
        # Supressão não-máxima: descarta candidatos muito próximos
        muito_proximo = False
        for _, sel_y, sel_x in selecionados:
            if abs(y - sel_y) < nhood_size and abs(x - sel_x) < nhood_size:
                muito_proximo = True
                break
        if not muito_proximo:
            selecionados.append((val, y, x))
    
    acc_vals = [s[0] for s in selecionados]
    centers_y = [s[1] for s in selecionados]
    centers_x = [s[2] for s in selecionados]
    
    return acc_vals, centers_y, centers_x

# Execução do código principal (como no notebook do professor)
if __name__ == '__main__':
    # 1. Carrega a imagem e converte para escala de cinza
    img = imread("image_0008.jpg")
    if img.ndim == 3:
        img_gray = rgb2gray(img)
    else:
        img_gray = img.astype(float)
    
    # 2. Exibe a imagem original
    plt.figure(figsize=(6,6))
    plt.imshow(img_gray, cmap='gray')
    plt.title("Imagem original")
    plt.axis("off")
    plt.show()
    
    # 3. Extração de bordas usando o filtro sobel
    edges = sobel(img_gray)
    plt.figure(figsize=(6,6))
    plt.imshow(edges, cmap='gray')
    plt.title("Bordas (Sobel)")
    plt.axis("off")
    plt.show()
    
    # 4. Binarização da imagem
    # Seguindo o exemplo do professor, utiliza-se:
    # limiar = image.max()*(10/256)
    limiar = img_gray.max() * (10/256)
    binary = edges.copy()
    binary[binary <= limiar] = 0
    binary[binary > 0] = 1
    plt.figure(figsize=(6,6))
    plt.imshow(binary*255, cmap='gray')
    plt.title("Imagem binária")
    plt.axis("off")
    plt.show()
    
    # 5. Operadores morfológicos (fechamento e abertura)
    binary = binary_closing(binary)
    binary = binary_opening(binary)
    plt.figure(figsize=(6,6))
    plt.imshow(binary, cmap='gray')
    plt.title("Imagem binária pós morfologia")
    plt.axis("off")
    plt.show()
    
    # 6. Aplicação da Transformada de Hough ingênua para um raio fixo
    raio_fixo = 15
    accumulator = naive_hough_circle(binary, raio_fixo)
    
    # 7. Exibe a grade de acumuladores para o raio fixo (com os anéis sobrepostos)
    plt.figure(figsize=(6,6))
    plt.imshow(accumulator, cmap='gray')
    plt.title("Grade de acumuladores para r = {}".format(raio_fixo))
    plt.axis("off")
    plt.show()
    
    # 8. Obtenção dos centros dos círculos (picos na grade do acumulador)
    total_num_peaks = 6
    acc_vals, centers_y, centers_x = naive_hough_circle_peaks(accumulator, total_num_peaks=total_num_peaks, nhood_size=15)
    
    # 9. Desenho das circunferências detectadas sobre a imagem original
    image_color = gray2rgb(img_gray)  # converte para RGB para desenhar colorido
    for center_y, center_x in zip(centers_y, centers_x):
        rr, cc = circle_perimeter(center_y, center_x, raio_fixo, shape=img_gray.shape)
        # Desenha a circunferência em azul (valores no intervalo 0..1)
        image_color[rr, cc] = (1, 0, 0)
    
    plt.figure(figsize=(6,6))
    plt.imshow(image_color)
    plt.title("Círculos detectados")
    plt.axis("off")
    plt.show()

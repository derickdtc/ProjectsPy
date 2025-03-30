#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.morphology import binary_opening, binary_closing
from skimage.draw import circle_perimeter
import math

def hough_circle_gradient(img_binaria, img_gray, raios):
    """
    Implementação otimizada da Transformada de Hough Circular usando a direção do gradiente
    para votar somente uma vez por pixel, para os raios especificados.
    
    Parâmetros:
    -----------
    img_binaria : ndarray
        Imagem binária (0 ou 1) dos pixels de borda.
    img_gray : ndarray
        Imagem em escala de cinza (float) para computar os gradientes.
    raios : iterable
        Lista com os raios a serem testados.
    
    Retorno:
    --------
    acumulador : ndarray com shape (len(raios), altura, largura)
        Acumulador 3D onde cada plano corresponde a um raio.
    """
    altura, largura = img_binaria.shape
    n_radii = len(raios)
    acumulador = np.zeros((n_radii, altura, largura), dtype=np.uint64)
    
    # Computa os gradientes da imagem em cinza
    gy, gx = np.gradient(img_gray)
    theta = np.arctan2(gy, gx)
    
    # Seleciona os pixels de borda
    edge_y, edge_x = np.nonzero(img_binaria)
    theta_edge = theta[edge_y, edge_x]
    
    # Converte os raios para array e obtém o número de pixels de borda
    raios_arr = np.array(raios)
    n_edges = edge_x.size

    # Para cada pixel de borda, vota no centro para todos os raios usando a direção do gradiente.
    # (a, b) = (x - r*cos(theta), y - r*sin(theta))
    candidate_a = np.rint(edge_x[:, None] - raios_arr[None, :] * np.cos(theta_edge[:, None])).astype(int)
    candidate_b = np.rint(edge_y[:, None] - raios_arr[None, :] * np.sin(theta_edge[:, None])).astype(int)
    
    # Cria um array com os índices dos raios para cada voto
    r_indices = np.tile(np.arange(n_radii), (n_edges, 1))
    
    # Achata os arrays para atualização vetorizada
    candidate_a = candidate_a.ravel()
    candidate_b = candidate_b.ravel()
    r_indices = r_indices.ravel()
    
    # Filtra os votos que caem dentro dos limites da imagem
    valid = (candidate_a >= 0) & (candidate_a < largura) & (candidate_b >= 0) & (candidate_b < altura)
    candidate_a = candidate_a[valid]
    candidate_b = candidate_b[valid]
    r_indices = r_indices[valid]
    
    # Atualiza o acumulador
    np.add.at(acumulador, (r_indices, candidate_b, candidate_a), 1)
    
    return acumulador

def hough_circle_peaks(hspaces, raios, total_num_peaks=10, threshold=None, nhood_size=8):
    """
    Localiza picos no acumulador 3D gerado pela CHT.
    
    Parâmetros:
    -----------
    hspaces : ndarray
        Acumulador 3D da CHT, com shape (len(raios), altura, largura).
    raios : iterable
        Lista de raios correspondentes à dimensão 0 de hspaces.
    total_num_peaks : int
        Número máximo de picos a serem retornados.
    threshold : float ou None
        Valor mínimo para ser considerado pico. Se None, usa 50% do valor máximo.
    nhood_size : int
        Tamanho da vizinhança para supressão não máxima (dentro de cada plano de raio).
    
    Retorno:
    --------
    acc_vals : list
        Valores do acumulador nos picos.
    centers_y : list
        Coordenadas Y dos centros detectados.
    centers_x : list
        Coordenadas X dos centros detectados.
    radii_encontrados : list
        Raios correspondentes a cada centro detectado.
    """
    if threshold is None:
        threshold = 0.5 * hspaces.max()
    
    candidatos = []
    for r_i in range(hspaces.shape[0]):
        plano = hspaces[r_i]
        ys, xs = np.nonzero(plano >= threshold)
        for y, x in zip(ys, xs):
            candidatos.append((plano[y, x], r_i, y, x))
    
    candidatos.sort(key=lambda c: c[0], reverse=True)
    
    picos_selecionados = []
    for val, r_i, cy, cx in candidatos:
        if len(picos_selecionados) >= total_num_peaks:
            break
        valido = True
        for _, r_i2, cy2, cx2 in picos_selecionados:
            if r_i == r_i2:
                if abs(cy - cy2) < nhood_size and abs(cx - cx2) < nhood_size:
                    valido = False
                    break
        if valido:
            picos_selecionados.append((val, r_i, cy, cx))
    
    acc_vals = [p[0] for p in picos_selecionados]
    r_idxs = [p[1] for p in picos_selecionados]
    centers_y = [p[2] for p in picos_selecionados]
    centers_x = [p[3] for p in picos_selecionados]
    radii_encontrados = [raios[r] for r in r_idxs]
    
    return acc_vals, centers_y, centers_x, radii_encontrados

if __name__ == "__main__":
    # 1) Carrega a imagem (certifique-se de que "imagemPIMG.jpeg" está no mesmo diretório)
    img = imread("image_0008.jpg")
    
    # Converte para escala de cinza se necessário
    if img.ndim == 3:
        img_gray = rgb2gray(img)
    else:
        img_gray = img.astype(float)
    
    # 2) Detecção de bordas com o filtro sobel
    edges = sobel(img_gray)
    
    # 3) Binarização: utiliza um limiar baseado na média dos valores dos bordos
    thresh = edges.mean()
    img_bin = edges > thresh
    
    # 4) Operações morfológicas para limpar a imagem binária
    img_bin = binary_closing(img_bin)
    img_bin = binary_opening(img_bin)
    
    # 5) Define o raio fixo (único) a ser testado: 10 pixels
    raio_fixo = 12
    raios = [12]
    
    # 6) Computa o acumulador usando a função otimizada (baseada em gradiente)
    acumulador = hough_circle_gradient(img_bin, img_gray, raios)
    
    # 7) Detecta os picos (centros dos círculos) no acumulador
    acc_vals, centers_y, centers_x, radii_encontrados = hough_circle_peaks(
        acumulador, raios,
        total_num_peaks=5,
        threshold=None,
        nhood_size=15
    )
    
    # 8) Prepara as imagens para exibição:
    #    - Converte a imagem original para RGB para desenhar os círculos.
    if img_gray.max() > 1:
        img_plot = img_gray / 255.0
    else:
        img_plot = img_gray
    img_rgb = np.dstack([img_plot, img_plot, img_plot])
    
    # Desenha os círculos detectados (em vermelho)
    for (cy, cx, r) in zip(centers_y, centers_x, radii_encontrados):
        rr, cc = circle_perimeter(cy, cx, r, shape=img_rgb.shape)
        img_rgb[rr, cc] = (1, 0, 0)
    
    # 9) Exibe os resultados:
    #     - Imagem binária pós morfologia
    #     - Grade do acumulador para o raio fixo (único plano)
    #     - Imagem original com os círculos detectados
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(img_bin, cmap="gray")
    ax[0].set_title("Imagem binária (pós morfologia)")
    ax[0].axis("off")
    
    ax[1].imshow(acumulador[0], cmap="jet")
    ax[1].set_title(f"Grade de acumuladores para r = {raio_fixo}")
    ax[1].axis("off")
    
    ax[2].imshow(img_rgb)
    ax[2].set_title("Círculos detectados (CHT otimizada)")
    ax[2].axis("off")
    
    plt.tight_layout()
    plt.show()

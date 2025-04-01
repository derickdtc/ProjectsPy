import numpy as np
import json
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def define_ground_truth(image_paths):
    gt_boxes = {}
    
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        gt_boxes[current_image] = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
        plt.close()
    
    for path in image_paths:
        global current_image
        current_image = path
        img = Image.open(path)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        rs = RectangleSelector(ax, onselect, useblit=True,
                               button=[1], minspanx=5, minspany=5,
                               spancoords='pixels', interactive=True)
        plt.title(f'Selecione o veículo: {path}')
        plt.show()
    
    with open('ground_truth.json', 'w') as f:
        json.dump(gt_boxes, f)

# Executar apenas uma vez para criar as anotações

# Converter para escala de cinza
def to_grayscale(img):
    return np.array(img.convert('L'), dtype=np.uint8)

def histogram_equalization(gray_np):
    # Calcula o histograma da imagem com 256 bins, correspondendo aos níveis de cinza
    hist, bins = np.histogram(gray_np.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    equalized = np.interp(gray_np.flatten(), bins[:-1], cdf_normalized)
    # Retorna a imagem equalizada com a mesma forma original e tipo uint8
    return equalized.reshape(gray_np.shape).astype(np.uint8)

def otsu_threshold(image):
    # Calcula o threshold ótimo utilizando o método de Otsu, que busca minimizar a variância intra-classe
    # Calcula o histograma da imagem
    hist, bin_edges = np.histogram(image.flatten(), bins=256, range=(0,256))
    total = image.size  # Número total de pixels na imagem
    # Calcula a soma ponderada dos valores de intensidade
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0  # Soma dos valores da classe de fundo
    wB = 0    # Número de pixels na classe de fundo
    var_max = 0  # Variância máxima entre as classes encontrada até agora
    threshold = 0  # Valor do threshold que maximiza a variância entre classes
    # Percorre todos os possíveis thresholds de 0 a 255
    for i in range(256):
        wB += hist[i]  # Incrementa o peso (número de pixels) da classe de fundo
        if wB == 0:
            continue
        wF = total - wB  # Número de pixels na classe do primeiro plano
        if wF == 0:
            break
        sumB += i * hist[i]  # Soma ponderada dos pixels da classe de fundo
        mB = sumB / wB  # Média da classe de fundo
        mF = (sum_total - sumB) / wF  # Média da classe do primeiro plano
        # Calcula a variância entre as classes
        var_between = wB * wF * (mB - mF) ** 2
        # Se a variância entre classes for maior, atualiza o threshold
        if var_between > var_max:
            var_max = var_between
            threshold = i
    return threshold

def adaptive_threshold_edges(edge_mag):
    # Aplica um threshold adaptativo à imagem de magnitude de bordas usando o método de Otsu
    thresh = otsu_threshold(edge_mag)
    # Cria uma imagem binária: pixels com valor igual ou superior ao threshold recebem 255, os demais recebem 0
    binary = (edge_mag >= thresh).astype(np.uint8) * 255
    return binary

# Filtro de mediana
def apply_median_filter(gray_np, kernel_size=3):
   
    pad = kernel_size // 2  # Calcula o padding necessário para manter o centro da janela
    h, w = gray_np.shape
    filtered = np.zeros_like(gray_np)
    # Percorre cada pixel (excetuando as bordas onde a janela não se encaixa)
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            # Seleciona a janela de pixels ao redor do pixel (i, j)
            window = gray_np[i-pad:i+pad+1, j-pad:j+pad+1]
            # Atribui a mediana dos valores da janela ao pixel central
            filtered[i, j] = np.median(window)
    return filtered

# Detecção de bordas (Sobel)
def sobel_edge_detection(gray_np):
    # Define os kernels para as direções x e y
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    
    h, w = gray_np.shape
    # Inicializa um array para armazenar a magnitude das bordas
    edge_mag = np.zeros_like(gray_np, dtype=np.float32)
    pad = 1  # Define o padding para garantir que a janela 3x3 caiba na região processada
    gray_float = gray_np.astype(np.float32)
    
    # Percorre os pixels ignorando as bordas para aplicar o kernel
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            # Extrai a região local em torno do pixel (i, j)
            region = gray_float[i-pad:i+pad+1, j-pad:j+pad+1]
            # Calcula a derivada na direção x e y
            gx = np.sum(region * Kx)
            gy = np.sum(region * Ky)
            # Calcula a magnitude da borda usando a norma Euclidiana
            edge_mag[i, j] = np.sqrt(gx**2 + gy**2)
    
    # Normaliza a magnitude das bordas para o intervalo [0,255]
    if edge_mag.max() != 0:
        edge_mag = (edge_mag / edge_mag.max()) * 255
    return np.clip(edge_mag, 0, 255).astype(np.uint8)

# Operações morfológicas
def erode(binary_np, se_size=3):
    pad = se_size // 2
    h, w = binary_np.shape
    out = np.zeros_like(binary_np)
    # Percorre cada pixel (exceto as bordas da imagem)
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            # Seleciona a janela definida pelo elemento estruturante
            window = binary_np[i-pad:i+pad+1, j-pad:j+pad+1]
            # Se todos os pixels da janela forem 255, o pixel central permanece 255
            if np.all(window == 255):
                out[i, j] = 255
    return out

def dilate(binary_np, se_size=3):
    pad = se_size // 2
    h, w = binary_np.shape
    out = np.zeros_like(binary_np)
    # Percorre cada pixel (excetuando as bordas onde a janela não se encaixa)
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            # Seleciona a janela do elemento estruturante
            window = binary_np[i-pad:i+pad+1, j-pad:j+pad+1]
            # Se pelo menos um pixel da janela for 255, o pixel central é definido como 255
            if np.any(window == 255):
                out[i, j] = 255
    return out

def close_binary(binary_np, se_size=5):
    return erode(dilate(binary_np, se_size), se_size)

def open_binary(binary_np, se_size=3):
    return dilate(erode(binary_np, se_size), se_size)

# Rotulagem de componentes conectados com conectividade de 8 vizinhos
def find_all_bounding_boxes(binary_np, min_size=500):
    # Identifica todas as regiões conectadas na imagem binária com área maior ou igual a min_size
    visited = np.zeros_like(binary_np, dtype=bool)  # Matriz para marcar pixels visitados
    h, w = binary_np.shape
    bboxes = [] 

    def neighbors(r, c):
        # Retorna os vizinhos do pixel (r, c) considerando conectividade de 8 vizinhos
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Ignora o próprio pixel
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

    # Percorre todos os pixels da imagem
    for i in range(h):
        for j in range(w):
            if binary_np[i, j] == 255 and not visited[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                pixels = []  # Lista para armazenar os pixels da região conectada
                # Busca em profundidade (DFS) para encontrar todos os pixels conectados
                while stack:
                    r, c = stack.pop()
                    pixels.append((r, c))
                    for nr, nc in neighbors(r, c):
                        if binary_np[nr, nc] == 255 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                # Após encontrar a região, calcula os limites (bounding box)
                if pixels:
                    r_coords = [p[0] for p in pixels]
                    c_coords = [p[1] for p in pixels]
                    r_min, r_max = min(r_coords), max(r_coords)
                    c_min, c_max = min(c_coords), max(c_coords)
                    # Calcula a área do bounding box
                    area = (r_max - r_min + 1) * (c_max - c_min + 1)
                    # Se a área é maior que o mínimo especificado, adiciona o bounding box à lista
                    if area >= min_size:
                        bboxes.append((c_min, r_min, c_max, r_max))
    return bboxes

def select_best_bbox(bboxes, img_w, img_h,
                     min_area_ratio,  # permite caixas menores
                     max_area_ratio,
                     min_ratio,
                     max_ratio):
    # Seleciona o bounding box mais plausível (possivelmente o carro) baseado em critérios de área e proporção
    img_area = img_w * img_h  # Área total da imagem
    candidates = []
    # Filtra os bounding boxes com base em área relativa e razão largura/altura
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        w_bbox = x_max - x_min + 1
        h_bbox = y_max - y_min + 1
        area = w_bbox * h_bbox
        ratio = w_bbox / float(h_bbox) if h_bbox != 0 else 9999

        # Verifica se a área do bounding box está entre os limites especificados
        if area < min_area_ratio * img_area or area > max_area_ratio * img_area:
            continue
        # Verifica se a proporção (razão largura/altura) está entre os limites permitidos
        if ratio < min_ratio or ratio > max_ratio:
            continue

        candidates.append(bbox)

    # Se nenhum candidato satisfaz os critérios, retorna None
    if not candidates:
        return None

    # Para depuração, seleciona o bounding box de maior área dentre os candidatos
    best_bbox = None
    best_area = 0
    for bbox in candidates:
        x_min, y_min, x_max, y_max = bbox
        area = (x_max - x_min + 1) * (y_max - y_min + 1)
        if area > best_area:
            best_area = area
            best_bbox = bbox
    return best_bbox
    
 # Função principal que processa a imagem e tenta detectar um veículo 

def calculate_iou(boxA, boxB):
    # Box format: (x_min, y_min, x_max, y_max)
    if boxA is None or boxB is None:
        return 0.0
    
    # Determinar coordenadas da interseção
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Calcular área de interseção
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Calcular áreas individuais
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Calcular IoU
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

def detect_vehicle(image_path):
    with open('ground_truth.json') as f:
        gt_boxes = json.load(f)
    pil_img = Image.open(image_path)  # Carrega a imagem utilizando a biblioteca PIL
    w, h = pil_img.size  # Obtém a largura e a altura da imagem

    gray = to_grayscale(pil_img) 
    gray = histogram_equalization(gray)  
    
    # Aplica o filtro de mediana
    med_filtered = apply_median_filter(gray, kernel_size=3)
    
    edges = sobel_edge_detection(med_filtered)  # Detecta as bordas usando o operador de Sobel
    edge_bin = adaptive_threshold_edges(edges)  # Binariza a imagem 
    
    closed = close_binary(edge_bin, se_size=15)
    opened = open_binary(closed, se_size=5)
    
    bboxes = find_all_bounding_boxes(opened, min_size=500)
    # Tenta selecionar o bounding box que melhor se encaixa nos critérios para ser considerado um carro
    best = select_best_bbox(bboxes, w, h,
                            min_area_ratio=0.1,
                            max_area_ratio=0.95,
                            min_ratio=1.2,
                            max_ratio=3.0)

    # Para depuração: desenha todos os bounding boxes encontrados em azul
    draw = ImageDraw.Draw(pil_img)
    for bbox in bboxes:
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="blue", width=1)
    
    if best is not None:
        detected_box = (best[0], best[1], best[2], best[3])
    else:
        detected_box = None
    
    gt_box = tuple(gt_boxes.get(image_path, (0,0,0,0)))
    iou = calculate_iou(detected_box, gt_box) if detected_box else 0.0

    draw.text((10, 10), f"IoU: {iou:.2f}", fill="yellow")
    # Se um bounding box plausível foi encontrado, destaca-o em vermelho e imprime as informações
    if best is not None:
        print(f"Imagem '{image_path}': Carro detectado!")
        print(f"BBox = (x_min={best[0]}, y_min={best[1]}, x_max={best[2]}, y_max={best[3]})")
        draw.rectangle([best[0], best[1], best[2], best[3]], outline="red", width=3)
        title = f"Carro detectado: {image_path}"
        print(f"IoU: {iou:.2f}")
    else:
        # Caso contrário, informa que nenhum carro foi detectado
        print(f"Imagem '{image_path}': NÃO há carro detectado.")
        title = f"Sem carro detectado: {image_path}"
    
    # Exibe a imagem com os bounding boxes desenhados
    plt.figure(figsize=(8,6))
    plt.imshow(pil_img)
    plt.title(title)
    plt.axis("off")
    plt.show()

def main():

    image_list = [
        "ProjectsPy\ImageProcessing\TrabalhoFinal\image_0042.jpg",
        "ProjectsPy\ImageProcessing\TrabalhoFinal\image_0014.jpg",
        "ProjectsPy\ImageProcessing\TrabalhoFinal\image_0028.jpg",  
        "ProjectsPy\ImageProcessing\TrabalhoFinal\image_0012.jpg",
        "ProjectsPy\ImageProcessing\TrabalhoFinal\image_0003.jpg",
        "ProjectsPy\ImageProcessing\TrabalhoFinal\image_0004.jpg",
        "ProjectsPy\ImageProcessing\TrabalhoFinal\image_0006.jpg",
        "ProjectsPy\ImageProcessing\TrabalhoFinal\image_0007.jpg",
        "ProjectsPy\ImageProcessing\TrabalhoFinal\image_0008.jpg",
        "ProjectsPy\ImageProcessing\TrabalhoFinal\image_0009.jpg",
    ]
    # define_ground_truth(image_list)  # Comente após criar o arquivo

    for path in image_list:
        detect_vehicle(path)

if __name__ == "__main__":
    main()

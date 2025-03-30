import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Converter para escala de cinza
def to_grayscale(img):
    return np.array(img.convert('L'), dtype=np.uint8)


def histogram_equalization(gray_np):
    hist, bins = np.histogram(gray_np.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    equalized = np.interp(gray_np.flatten(), bins[:-1], cdf_normalized)
    return equalized.reshape(gray_np.shape).astype(np.uint8)


#  Filtro de mediana
def apply_median_filter(gray_np, kernel_size=3):
    pad = kernel_size // 2
    h, w = gray_np.shape
    filtered = np.zeros_like(gray_np)
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            window = gray_np[i-pad:i+pad+1, j-pad:j+pad+1]
            filtered[i, j] = np.median(window)
    return filtered


#  Detecção de bordas (Sobel)
def sobel_edge_detection(gray_np):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    
    h, w = gray_np.shape
    edge_mag = np.zeros_like(gray_np, dtype=np.float32)
    pad = 1
    gray_float = gray_np.astype(np.float32)
    
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            region = gray_float[i-pad:i+pad+1, j-pad:j+pad+1]
            gx = np.sum(region * Kx)
            gy = np.sum(region * Ky)
            edge_mag[i, j] = np.sqrt(gx**2 + gy**2)
    
    if edge_mag.max() != 0:
        edge_mag = (edge_mag / edge_mag.max()) * 255
    return np.clip(edge_mag, 0, 255).astype(np.uint8)


#  Limiarização simples
def threshold_edges(edge_mag, thresh=30):
    """
    Binariza a imagem de bordas.
    'thresh' deve ser relativamente baixo para tentar
    capturar mais bordas do carro.
    """
    binary = (edge_mag >= thresh).astype(np.uint8) * 255
    return binary

# Morfologia Binária (erosão, dilatação)

def erode(binary_np, se_size=3):
    pad = se_size // 2
    h, w = binary_np.shape
    out = np.zeros_like(binary_np)
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            window = binary_np[i-pad:i+pad+1, j-pad:j+pad+1]
            if np.all(window == 255):
                out[i, j] = 255
    return out

def dilate(binary_np, se_size=3):
    pad = se_size // 2
    h, w = binary_np.shape
    out = np.zeros_like(binary_np)
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            window = binary_np[i-pad:i+pad+1, j-pad:j+pad+1]
            if np.any(window == 255):
                out[i, j] = 255
    return out

def close_binary(binary_np, se_size=7):
    """
    Fechamento com elemento estruturante maior (7)
    para unir bordas e preencher lacunas.
    """
    return erode(dilate(binary_np, se_size), se_size)

def open_binary(binary_np, se_size=5):
    """
    Abertura com elemento intermediário (5) para remover ruídos.
    """
    return dilate(erode(binary_np, se_size), se_size)

#  Rotulagem de todos os componentes

def find_all_bounding_boxes(binary_np, min_size=2000):
    """
    Retorna lista de bounding boxes (x_min, y_min, x_max, y_max)
    cujas áreas sejam >= min_size.
    """
    visited = np.zeros_like(binary_np, dtype=bool)
    h, w = binary_np.shape
    bboxes = []

    def neighbors(r, c):
        for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc

    for i in range(h):
        for j in range(w):
            if binary_np[i, j] == 255 and not visited[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                pixels = []
                while stack:
                    r, c = stack.pop()
                    pixels.append((r, c))
                    for nr, nc in neighbors(r, c):
                        if binary_np[nr, nc] == 255 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                if len(pixels) > 0:
                    r_coords = [p[0] for p in pixels]
                    c_coords = [p[1] for p in pixels]
                    r_min, r_max = min(r_coords), max(r_coords)
                    c_min, c_max = min(c_coords), max(c_coords)
                    area = (r_max - r_min + 1) * (c_max - c_min + 1)
                    if area >= min_size:
                        bboxes.append((c_min, r_min, c_max, r_max))
    return bboxes

#Tentando  selecionar bounding box plausível
def select_best_bbox(bboxes, img_w, img_h,
                     min_area_ratio=0.02,  # permitir área maior que 2% da imagem
                     max_area_ratio=0.8,   # até 80% da imagem
                     min_ratio=0.8,       # permitir mais variação
                     max_ratio=4.0):
    """
    Seleciona a bounding box mais plausível como 'carro'.
    É necessário o  ajuste dos limites conforme a escala do carro nas imagens.
    """
    img_area = img_w * img_h
    candidates = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        area = w * h
        ratio = w / float(h) if h != 0 else 9999

        if area < min_area_ratio * img_area or area > max_area_ratio * img_area:
            continue
        if ratio < min_ratio or ratio > max_ratio:
            continue

        # Se passou nos critérios básicos, adiciona
        candidates.append(bbox)

    if not candidates:
        return None

    # Se houver mais de um candidato, escolhe o de maior área
    best_bbox = None
    best_area = 0
    for bbox in candidates:
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        area = w * h
        if area > best_area:
            best_area = area
            best_bbox = bbox
    return best_bbox


def detect_vehicle(image_path):
    """
    Tenta detectar o carro na imagem, marcando um bounding box maior
    que englobe o carro, usando operações morfológicas e threshold de bordas
    """
    #  Carrega a imagem
    pil_img = Image.open(image_path)
    w, h = pil_img.size

    #  Converte para cinza
    gray = to_grayscale(pil_img)

    #  Filtro de mediana
    med_filtered = apply_median_filter(gray, kernel_size=5)

    #  Sobel
    edges = sobel_edge_detection(med_filtered)

    #  Threshold das bordas (mais baixo)
    edge_bin = threshold_edges(edges, thresh=30)

    closed = close_binary(edge_bin, se_size=7)
    opened = open_binary(closed, se_size=5)

    #  Rotulagem e obtenção de bounding boxes
    bboxes = find_all_bounding_boxes(opened, min_size=2000)

    best = select_best_bbox(bboxes, w, h,
                            min_area_ratio=0.02,
                            max_area_ratio=0.8,
                            min_ratio=0.8,
                            max_ratio=4.0)

    #  Plotar resultado
    if best is not None:
        print(f"Imagem '{image_path}': Carro detectado!")
        print(f"BBox = (x_min={best[0]}, y_min={best[1]}, x_max={best[2]}, y_max={best[3]})")
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([best[0], best[1], best[2], best[3]], outline="red", width=3)
        plt.figure(figsize=(8,6))
        plt.imshow(pil_img)
        plt.title(f"Carro detectado: {image_path}")
        plt.axis("off")
        plt.show()
    else:
        print(f"Imagem '{image_path}': NÃO há carro detectado.")
        plt.figure(figsize=(8,6))
        plt.imshow(pil_img)
        plt.title(f"Sem carro detectado: {image_path}")
        plt.axis("off")
        plt.show()

def main():
    image_list = [
        "naoECarro/imagemPIMG.jpeg",
        "naoECarro/imagemPIMG2.jpeg",
        "naoECarro/imagemPIMG3.jpeg",
        "image_0042.jpg",
        "image_0014.jpg",
        "image_0028.jpg",  
        "image_0012.jpg",
        "image_0003.jpg",
        "image_0004.jpg",
        "image_0006.jpg",
        "image_0007.jpg",
        "image_0008.jpg",
        "image_0009.jpg",
    ]
    for path in image_list:
        detect_vehicle(path)

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure, filters, morphology, measure

def load_image_rgb(path):
    """
    Carrega a imagem e converte para RGB (descartando canal alfa se existir).
    Retorna um array [H, W, 3].
    """
    image = io.imread(path)
    if image.ndim == 3 and image.shape[2] == 4:
        # Remove canal alfa (RGBA -> RGB)
        image = image[:, :, :3]
    return image

def preprocess_and_gray(image_rgb):
    """
    Converte a imagem para tons de cinza e aplica equalização adaptativa 
    para melhorar o contraste, retornando array [0..1].
    """
    # Converte para escala de cinza [0..1]
    gray = color.rgb2gray(image_rgb)
    # Equalização adaptativa de histograma (CLAHE)
    gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)
    return gray_eq

def segment_image(gray_image):
    """
    Segmenta a imagem usando o limiar de Otsu e aplica morfologia
    para remover ruídos e unir partes.
    """
    # Limiar automático (Otsu)
    thresh = filters.threshold_otsu(gray_image)
    binary = gray_image > thresh

    # Remove pequenos objetos (ruídos) - ajuste min_size conforme necessário
    cleaned = morphology.remove_small_objects(binary, min_size=800)

    # Fechamento morfológico para preencher falhas dentro do carro
    closed = morphology.closing(cleaned, morphology.square(5))

    return closed

def find_vehicle_regions(binary, min_area=5000, max_area_ratio=0.5,
                        min_solidity=0.3, aspect_ratio_range=(0.8, 4.0)):
    """
    Retorna uma lista de (region, bbox) que atendam aos critérios de filtragem:
      - Área mínima > min_area e < (max_area_ratio * área total)
      - Solidez acima de min_solidity
      - Razão de aspecto (width/height) dentro de aspect_ratio_range
    """
    labels = measure.label(binary)
    props = measure.regionprops(labels)
    H, W = binary.shape
    total_area = H * W

    candidates = []
    for region in props:
        area = region.area
        if area < min_area or area > max_area_ratio * total_area:
            continue
        if region.solidity < min_solidity:
            continue

        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        if height == 0:
            continue
        aspect_ratio = width / height

        if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
            continue

        # Se passou pelos critérios, adiciona como candidato
        candidates.append((region, (minr, minc, maxr, maxc)))

    return candidates

def detect_car_bbox(image_rgb):
    """
    Retorna a bounding box (minr, minc, maxr, maxc) do carro encontrado
    ou None se não encontrar nada.
    """
    # 1. Converte para cinza + equalização adaptativa
    gray_eq = preprocess_and_gray(image_rgb)

    # 2. Segmenta (binariza + morfologia)
    binary = segment_image(gray_eq)

    # 3. Busca regiões candidatas a veículo
    candidates = find_vehicle_regions(
        binary,
        min_area=5000,       # Ajuste conforme suas imagens
        max_area_ratio=0.6,
        min_solidity=0.3,
        aspect_ratio_range=(0.8, 4.0)
    )

    if not candidates:
        return None

    # 4. Escolhe a maior região em área (assumindo que o carro seja a maior)
    best_region = None
    best_area = 0
    for region, bbox in candidates:
        if region.area > best_area:
            best_area = region.area
            best_region = (region, bbox)

    if best_region is None:
        return None

    # Retorna a bounding box final
    return best_region[1]  # (minr, minc, maxr, maxc)

def main():
    # Lista com as 10 imagens (exemplo)
    # Ajuste os nomes/caminhos conforme a sua organização
    image_list = [
        "carro1.png",
        "carro2.png",
        "carro3.png",
        "carro4.png",
        "carro5.png",
        "carro6.png",
        "carro7.png",
        "carro8.png",
        "carro9.png",
        "carro10.png"
    ]

    for path in image_list:
        print(f"\nProcessando imagem: {path}")
        image_rgb = load_image_rgb(path)
        bbox = detect_car_bbox(image_rgb)

        if bbox is None:
            print("  -> Nenhum veículo detectado!")
        else:
            print("  -> Veículo detectado!")
            (minr, minc, maxr, maxc) = bbox
            print(f"     Bounding Box = (minr={minr}, minc={minc}, maxr={maxr}, maxc={maxc})")

            # Exibir a imagem com bounding box
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.imshow(image_rgb)
            rect = plt.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor='red',
                linewidth=2
            )
            ax.add_patch(rect)
            ax.set_title(f"Veículo detectado em {path}")
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    main()

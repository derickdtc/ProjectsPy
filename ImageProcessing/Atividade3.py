import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte

# %% Funções Morfológicas Implementadas "na mão"

def binary_erosion(image, se):
    """
    Realiza a erosão binária de 'image' usando o elemento estruturante 'se'.
    image: array binário (0 e 1)
    se: elemento estruturante (array com 0 e 1)
    """
    se_h, se_w = se.shape
    pad_h = se_h // 2
    pad_w = se_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    eroded = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+se_h, j:j+se_w]
            if np.all(region[se == 1] == 1):
                eroded[i, j] = 1
    return eroded

def binary_dilation(image, se):
    """
    Realiza a dilatação binária de 'image' usando o elemento estruturante 'se'.
    """
    se_h, se_w = se.shape
    pad_h = se_h // 2
    pad_w = se_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    dilated = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+se_h, j:j+se_w]
            if np.any(region[se == 1] == 1):
                dilated[i, j] = 1
    return dilated

def binary_opening(image, se):
    """
    Realiza a abertura binária: erosão seguida de dilatação.
    """
    eroded = binary_erosion(image, se)
    opened = binary_dilation(eroded, se)
    return opened

# %% Esqueletização Morfológica

def morphological_skeleton(image, se):
    """
    Calcula o esqueleto morfológico da imagem binária 'image' usando o elemento estruturante 'se'.
    
    A ideia é iterativamente erodir a imagem e, a cada iteração, calcular:
        S_k = (A ⊖ kB) - ((A ⊖ kB) ∘ B)
    e acumular os S_k até que a imagem erodida fique vazia.
    """
    img_current = image.copy()
    skeleton = np.zeros_like(image)
    iteration = 0
    
    while np.any(img_current):
        eroded = binary_erosion(img_current, se)
        opened = binary_dilation(eroded, se)
        S_k = np.logical_and(img_current, np.logical_not(opened)).astype(np.uint8)
        skeleton = np.logical_or(skeleton, S_k).astype(np.uint8)
        img_current = eroded.copy()
        iteration += 1
        if iteration > 1000:
            break
    return skeleton

# %% Funções Auxiliares para Poda Refinada

def find_endpoints(skel):
    """
    Identifica os pontos finais (endpoints) na imagem esquelética.
    Um pixel é endpoint se ele é 1 e tem apenas 1 vizinho na vizinhança 8-conectada.
    """
    endpoints = np.zeros_like(skel)
    for i in range(1, skel.shape[0]-1):
        for j in range(1, skel.shape[1]-1):
            if skel[i, j] == 1:
                neighborhood = skel[i-1:i+2, j-1:j+2]
                if (np.sum(neighborhood) - 1) == 1:
                    endpoints[i, j] = 1
    return endpoints

def follow_branch(skel, start, max_length):
    """
    A partir de um endpoint 'start', segue a ramificação enquanto:
      - houver exatamente um vizinho (exceto o pixel anterior)
      - o comprimento não ultrapassar 'max_length'
      - não for encontrada uma junção (mais de 1 vizinho)
      
    Retorna a lista de coordenadas que compõem o ramo.
    """
    branch = [start]
    current = start
    prev = None
    while True:
        i, j = current
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                # Verifica limites da imagem
                if ni < 0 or ni >= skel.shape[0] or nj < 0 or nj >= skel.shape[1]:
                    continue
                if (ni, nj) == prev:
                    continue
                if skel[ni, nj] == 1:
                    neighbors.append((ni, nj))
        if len(neighbors) == 0:
            break
        if len(neighbors) > 1:
            # Encontrou uma junção; para o rastreamento
            break
        next_pixel = neighbors[0]
        branch.append(next_pixel)
        if len(branch) >= max_length:
            break
        prev = current
        current = next_pixel
    return branch

def refined_prune_skeleton(skel, branch_length_threshold=5):
    """
    Remove ramos espúrios da imagem esquelética.
    
    Em vez de remover iterativamente todos os endpoints, esta função
    segue o ramo a partir de cada endpoint e, se o comprimento do ramo for menor
    ou igual ao limiar definido (branch_length_threshold), remove-o.
    
    Isso evita que partes significativas da digital sejam removidas.
    """
    skel_pruned = skel.copy()
    endpoints = find_endpoints(skel_pruned)
    endpoint_coords = np.argwhere(endpoints == 1)
    for coord in endpoint_coords:
        i, j = coord
        branch = follow_branch(skel_pruned, (i, j), branch_length_threshold + 1)
        if len(branch) <= branch_length_threshold:
            for (p, q) in branch:
                skel_pruned[p, q] = 0
    return skel_pruned

# %% Processamento da Imagem

# Carrega a imagem (certifique-se de que "digital.png" esteja no mesmo diretório)
img = io.imread('digital.png')

# Converte para escala de cinza, se necessário
if img.ndim == 3:
    img_gray = color.rgb2gray(img)
else:
    img_gray = img

# Converte para 8-bit e normaliza para [0, 1]
img_gray = img_as_ubyte(img_gray)
img_norm = img_gray / 255.0

# Binarização: ajuste o threshold conforme necessário
threshold = 0.5
img_bin = (img_norm > threshold).astype(np.uint8)

# %% Escolha do Elemento Estruturante
se = np.ones((3, 3), dtype=np.uint8)

# %% Esqueletização
skeleton = morphological_skeleton(img_bin, se)

# %% Poda Refinada
# O parâmetro branch_length_threshold pode ser ajustado; valores pequenos (ex.: 5)
# removem apenas ramos muito curtos, preservando a estrutura principal.
pruned_skeleton = refined_prune_skeleton(skeleton, branch_length_threshold=5)

# %% Exibição das 4 imagens lado a lado
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title('Imagem Original\n(Escala de Cinza)')
axes[0].axis('off')

axes[1].imshow(img_bin, cmap='gray')
axes[1].set_title('Imagem Binarizada')
axes[1].axis('off')

axes[2].imshow(skeleton, cmap='gray')
axes[2].set_title('Esqueletização')
axes[2].axis('off')

axes[3].imshow(pruned_skeleton, cmap='gray')
axes[3].set_title('Esqueleto Após Poda Refinada')
axes[3].axis('off')

plt.tight_layout()
plt.show()

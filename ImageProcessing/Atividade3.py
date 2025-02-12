# Importa bibliotecas necessárias
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
    # Preenche com zeros (background)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    eroded = np.zeros_like(image)
    
    # Percorre todos os pixels da imagem original
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Região vizinha (mesmo tamanho do se)
            region = padded[i:i+se_h, j:j+se_w]
            # Se TODOS os pixels da região onde se==1 forem 1, mantém o pixel
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
            # Se QUALQUER pixel na região onde se==1 for 1, coloca 1
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
    # Cópia para não alterar a imagem original
    img_current = image.copy()
    skeleton = np.zeros_like(image)
    iteration = 0
    
    while np.any(img_current):  # enquanto houver pixel de foreground
        eroded = binary_erosion(img_current, se)
        opened = binary_dilation(eroded, se)
        # S_k = pixels que ainda estão na imagem, mas foram "removidos" pela abertura
        S_k = np.logical_and(img_current, np.logical_not(opened)).astype(np.uint8)
        skeleton = np.logical_or(skeleton, S_k).astype(np.uint8)
        img_current = eroded.copy()
        iteration += 1
        # Opção: limitar número de iterações para evitar laço infinito (caso necessário)
        if iteration > 1000:
            break
    return skeleton

# %% Algoritmo de Pruning

def find_endpoints(skel):
    """
    Identifica os pontos finais (endpoints) na imagem esquelética.
    Um pixel é endpoint se ele é 1 e tem apenas 1 vizinho em sua vizinhança 8-conectada.
    """
    endpoints = np.zeros_like(skel)
    # Evita borda (ou pode-se tratar a borda separadamente)
    for i in range(1, skel.shape[0]-1):
        for j in range(1, skel.shape[1]-1):
            if skel[i, j] == 1:
                # Considera os 8 vizinhos
                neighborhood = skel[i-1:i+2, j-1:j+2]
                # Exclui o pixel central e conta os pixels ativos
                if (np.sum(neighborhood) - 1) == 1:
                    endpoints[i, j] = 1
    return endpoints

def prune_skeleton(skel, iterations=10):
    """
    Remove ramos espúrios da imagem esquelética, removendo iterativamente os endpoints.
    
    Parâmetro 'iterations' define quantas iterações de remoção de endpoints serão realizadas.
    """
    pruned = skel.copy()
    for it in range(iterations):
        endpoints = find_endpoints(pruned)
        # Se não houver endpoints, encerra
        if np.sum(endpoints) == 0:
            break
        pruned[endpoints == 1] = 0
    return pruned

# %% Processamento da Imagem

# Carrega a imagem (certifique-se de que "digital.png" esteja no mesmo diretório do Notebook)
img = io.imread('digital.png')

# Se a imagem estiver em RGB, converte para escala de cinza
if img.ndim == 3:
    img_gray = color.rgb2gray(img)
else:
    img_gray = img

# Converte para 8-bit e normaliza para [0, 1]
img_gray = img_as_ubyte(img_gray)
img_norm = img_gray / 255.0

# Binarização: escolha um limiar adequado (por exemplo, 0.5 ou outro valor dependendo da imagem)
threshold = 0.5
img_bin = (img_norm > threshold).astype(np.uint8)

# Exibe a imagem original e a binarizada
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img_gray, cmap='gray')
ax[0].set_title('Imagem Original (Escala de Cinza)')
ax[0].axis('off')

ax[1].imshow(img_bin, cmap='gray')
ax[1].set_title('Imagem Binarizada')
ax[1].axis('off')

plt.show()

# %% Escolha do Elemento Estruturante

# Usaremos um elemento estruturante 3x3 com todos os pixels iguais a 1.
se = np.ones((3, 3), dtype=np.uint8)

# %% Esqueletização

skeleton = morphological_skeleton(img_bin, se)

plt.figure(figsize=(6,6))
plt.imshow(skeleton, cmap='gray')
plt.title('Esqueletização da Impressão Digital')
plt.axis('off')
plt.show()

# %% Pruning

# Remover os ramos espúrios - experimente variar o número de iterações
pruned_skeleton = prune_skeleton(skeleton, iterations=10)

plt.figure(figsize=(6,6))
plt.imshow(pruned_skeleton, cmap='gray')
plt.title('Esqueleto Após Pruning')
plt.axis('off')
plt.show()

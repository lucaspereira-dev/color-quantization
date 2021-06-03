import sys
sys.path.insert(0, './minisom')
from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# %matplotlib inline

# leia a imagem
img = plt.imread('./img/ogrito.jpg')

# remodelando a matriz de pixels
pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))

# Inicialização e treinamento do SOM
print('Treinamento...')
som = MiniSom(3, 3, 3, sigma=1., learning_rate=0.2, neighborhood_function='bubble')  # 3x3 = 9 final colors
som.random_weights_init(pixels)
starting_weights = som.get_weights().copy()   # salvando os pesos iniciais
som.train_random(pixels, 500)

print('Quantização...')
qnt = som.quantization(pixels)  # quantize cada pixel da imagem
print('Construindo nova imagem...')
clustered = np.zeros(img.shape)
for i, q in enumerate(qnt):  # coloque os valores quantizados em uma nova imagem
    clustered[np.unravel_index(i, dims=(img.shape[0], img.shape[1]))] = q
print('Feito.')

# Exibindo resultado
plt.figure(figsize=(7, 7))
plt.figure(1)
plt.subplot(221)
plt.title('Original')
plt.imshow(img)
plt.subplot(222)
plt.title('Resultado')
plt.imshow(clustered)

plt.subplot(223)
plt.title('Cores iniciais')
plt.imshow(starting_weights, interpolation='none')
plt.subplot(224)
plt.title('Cores aprendidas')
plt.imshow(som.get_weights(), interpolation='none')

plt.tight_layout()
plt.show()
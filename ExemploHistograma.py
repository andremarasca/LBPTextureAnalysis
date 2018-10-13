import matplotlib.image as mpimg
import numpy as np

# Carregar a imagem de entrada
imagem = mpimg.imread('ImagemTeste.png')
# Descobrir dimensões da imagem
forma = imagem.shape
# A Entrada ta no intervalo [0,1]
# Para trabalhar com histograma deve ser [0, 255]
imagem = imagem * 255
# Converter a imagem em uint8
imagem = imagem.astype(np.uint8)

# Inicializa o vetor de caracteristicas com 0
# O vetor de caracteristica nesse caso é o histograma
# da imagem de entrada, 256 cores possiveis
hist = np.zeros(256)

# calcular o histograma consiste em contar o número
# de pixels de cada cor [0, 255]...
# Para isso deve iterar os pixels da imagem contando
for k in range(forma[2]):
    for i in range(forma[0]):
        for j in range(forma[1]):
            hist[imagem[i][j][k]] += 1
 
# Plotar o histograma da imagem
import matplotlib.pyplot as plt
plt.plot(hist)
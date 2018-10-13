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
    for i in range(1, forma[0] -1):
        for j in range(1, forma[1]-1):
            a = imagem[i][j] >= imagem[i-1][j-1]
            b = imagem[i][j] >= imagem[i-1][j]
            c = imagem[i][j] >= imagem[i-1][j+1]
            d = imagem[i][j] >= imagem[i][j-1]
            e = imagem[i][j] >= imagem[i][j+1]
            f = imagem[i][j] >= imagem[i+1][j-1]
            g = imagem[i][j] >= imagem[i+1][j]
            h = imagem[i][j] >= imagem[i+1][j+1]
            res = a * 1 + b * 2 + c * 4 + d * 8 + e * 16 + f * 32 + g * 64 + h * 128
            hist[res] += 1
 
# Plotar o histograma da imagem
import matplotlib.pyplot as plt
plt.plot(hist)
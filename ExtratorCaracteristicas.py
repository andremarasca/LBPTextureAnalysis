import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
import os as osfnc
import numpy as np

#%% Descobrir o nome de todas as imagens da base

# Para isso é necessário descobrir o diretório atual
# Em seguida apontar a pasta da base de imagens
# Descobrir o nome de todos os arquivos da pasta
# Filtrar apenas aqueles que são imagens e adicionar
# Na lista de imagens

# Descobrir o diretório atual
mypath = osfnc.getcwd()
# apontar a pasta que as minhas imagens estão
mypath = mypath + '\\Base'
# Criar uma lista de imagens
soimagens = []
# Iterar cada um dos arquivos dentro de mypath
for f in listdir(mypath):
    #Calcular o diretório completo do arquivo
    aa = join(mypath, f)
    # verificar se é um arquivo ou não
    if isfile(aa):
        # Se for arquivo verificar se é .png
        if f.endswith(".png"):
            # Se for .png então adiciona na lista de imagens
            soimagens.append(aa)

#%%

# Inicializar o dataset zerado, já inicializar uma matriz
# O numero de linhas é a quantidade de imagens
# O numero de colunas é 256 quando 
# o vetor de características é o histograma
histogramas = np.zeros((len(soimagens), 256))

# Para cada uma das imagens calcular o histograma
for u, nome_das_imagens in enumerate(soimagens):
    # Abrir a imagem
    imagem = mpimg.imread(nome_das_imagens)
    # Descobrir dimensões da imagem
    forma = imagem.shape
    # A Entrada ta no intervalo [0,1]
    # Para trabalhar com histograma deve ser [0, 255]
    imagem = imagem * 255
    # Converter a imagem em uint8
    imagem = imagem.astype(np.uint8)
      
    print(u)
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
                histogramas[u, res] += 1
# Para o dataset do Pandas precisamos de nome nas colunas
lista_nomes = ['cor%03d' %i for i in range(256)]

# Transformar a matriz em dataset do pandas
import pandas as pd
base = pd.DataFrame(histogramas, columns=lista_nomes)

#%% inserir classe no dataset

# as primeiras 20 sao de laranja as outras 20 é de limão
lista_classes = ['laranja' for i in range(20)]
lista_classes.extend('limao' for i in range(20))

base['classe'] = lista_classes
# Salvar o o Dataset
base.to_csv('base.csv')

#%% Aprendizagem

# instânciar a rede neural
from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(max_iter = 1000, tol = 0.0001)

# Separar o dataset em previsores e classe
classe = base.pop(base.keys()[-1])
previsores = base.values

# Instânciar o Label Encoder da classe
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Calcular o label encoder da classe 
encoder_classe = labelencoder.fit(classe)
# Aplicar o label encoder na classe
classe = encoder_classe.transform(classe)

# Instânciar a padronização z-score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Calcular os parâmetros da z-score
scaler_previsores = scaler.fit(previsores)
# Aplicar a z-score na base
previsores = scaler_previsores.transform(previsores)

# Ensinar a rede neural a diferença entre laranja e limão
classificador.fit(previsores, classe)

#%% Salvar no disco os objetos para usar depois

import pickle

#Salvar o classificador no disco
pickle.dump(classificador, open('reconhecedor_laranjas_limoes.sav', 'wb'))
#Salvar o encoder da classe no disco
pickle.dump(encoder_classe, open('encoder_classe.sav', 'wb'))
#Salvar a padronização z-score no disco
pickle.dump(scaler_previsores, open('scaler_previsores.sav', 'wb'))
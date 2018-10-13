import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
import os as osfnc
import numpy as np

mypath = osfnc.getcwd()
mypath = mypath + '\\Testes' # Agora classificando a base de testes
soimagens = []
nomes_imagens = []
for f in listdir(mypath):
    aa = join(mypath, f)
    if isfile(aa):
        if f.endswith(".png"):
            soimagens.append(aa)
            nomes_imagens.append(f)

#%% Calcular histograma das imagens de teste

histogramas = np.zeros((len(soimagens), 256))

for u, nome_das_imagens in enumerate(soimagens):
    imagem = mpimg.imread(nome_das_imagens)
    forma = imagem.shape
    imagem = imagem * 255
    imagem = imagem.astype(np.uint8)
      
    print(u)
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
    
    
lista_nomes = ['cor%03d' %i for i in range(256)]

import pandas as pd
base = pd.DataFrame(histogramas, columns=lista_nomes)

#%% Uitlizar os modelos já existentes

import pickle

# Já salvei no disco os objetos
# não preciso importar as classes denovo
classificador = pickle.load(open('reconhecedor_laranjas_limoes.sav', 'rb'))
encoder_classe = pickle.load(open('encoder_classe.sav', 'rb'))
scaler_previsores = pickle.load(open('scaler_previsores.sav', 'rb'))
#Importado Rede Neural
#Importado encode da classe
#Importado z-score

# Padronizar a base de testes usando o z-score calculado anteriormente
base_padronizada = scaler_previsores.transform(base)

# Precição das instâncias de teste
predicao = classificador.predict(base_padronizada)

# Descobrir o nome das classes
predicao_nomes = encoder_classe.inverse_transform(predicao)

# Imprimir resultado
for classificacao, nome_original  in zip(predicao_nomes, nomes_imagens):
    print(classificacao + ' ' + nome_original)
    
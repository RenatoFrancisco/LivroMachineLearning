import pandas as pd
from collections import Counter

df = pd.read_csv('situacao_do_cliente.csv')
x_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
y_df = df['situacao']

Xdummies_df = pd.get_dummies(x_df).astype(int)
Ydummies_df = y_df

X = Xdummies_df.values
Y = Xdummies_df.values

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
# tamanho_de_validacao = len(y) - tamanho_de_treino - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

modelo = OneVsRestClassifier(LinearSVC(random_state = 0))
k = 3
scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k)
print(scores)

import numpy as np
media = np.mean(scores)
print(media)



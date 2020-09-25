import pandas as pd
from collections import Counter

df = pd.read_csv('buscas2.csv')
x_df = df[['home', 'busca', 'logado']]
y_df = df['comprou']
xdummies_df = pd.get_dummies(x_df)
ydummies_df = y_df

x = xdummies_df.values
y = ydummies_df.values

acerto_base = max(Counter(y).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(y)
print('Taxa de acerto base: %.2f' % taxa_de_acerto_base)

porcetagem_treino = 0.9

tamanho_de_treino = int(porcetagem_treino * len(y))
tamanho_de_teste = len(y) - tamanho_de_treino

treino_dados = x[:tamanho_de_treino]
treino_marcacoes = y[:tamanho_de_treino]

teste_dados = x[-tamanho_de_teste:]
teste_marcacoes = y[-tamanho_de_teste:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

acertos = resultado == teste_marcacoes
total_de_acertos = sum(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print('Taxa de acerto do algoritmo: %.2f' % taxa_de_acerto)
print(total_de_elementos)
import pandas as pd

df = pd.read_csv('buscas.csv')
x_df = df[['home', 'busca', 'logado']]
y_df = df['comprou']
xdummies_df = pd.get_dummies(x_df)
ydummies_df = y_df

x = xdummies_df.values
y = ydummies_df.values

acerto_de_um = len(y[y==1])
acerto_de_zero = len(y[y==0])
taxa_de_acerto_base = 100.0 * max(acerto_de_um, acerto_de_zero) / len(y)
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
diferencas = resultado - teste_marcacoes
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print('Taxa de acerto do algoritmo: %.2f' % taxa_de_acerto)
print(total_de_elementos)
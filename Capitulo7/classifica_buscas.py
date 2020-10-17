import pandas as pd
from collections import Counter

df = pd.read_csv('buscas2.csv')
x_df = df[['home', 'busca', 'logado']]
y_df = df['comprou']

xdummies_df = pd.get_dummies(x_df)
ydummies_df = y_df

x = xdummies_df.values
y = ydummies_df.values

porcetagem_de_treino = 0.8
porcetagem_de_teste = 0.1

tamanho_de_treino = int(porcetagem_de_treino * len(y))
tamanho_de_teste = int(porcetagem_de_teste * len(y))
tamanho_de_validacao = len(y) - tamanho_de_treino - tamanho_de_teste

treino_dados = x[:tamanho_de_treino]
treino_marcacoes = y[:tamanho_de_treino]

fim_de_treino = tamanho_de_treino + tamanho_de_teste

teste_dados = x[tamanho_de_treino:fim_de_treino]
teste_marcacoes = y[tamanho_de_treino:fim_de_treino]

validacao_dados = x[fim_de_treino:]
validacao_marcacoes = y[fim_de_treino:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)

    acertos = resultado == teste_marcacoes
    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    
    return taxa_de_acerto

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do vencedor entre os dois algoritomos no mundo real: {0}".format(taxa_de_acerto)
    print(msg)

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict('MultinomialNB', modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modelAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modelAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultadoMultinomial > resultadoAdaBoost:
    vencedor = modeloMultinomial
else:
    vencedor = modelAdaBoost

teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)
print('Taxa de acerto base: %.2f' % taxa_de_acerto_base)

total_de_elementos = len(teste_dados)
print('Total de teste: %d' % total_de_elementos)
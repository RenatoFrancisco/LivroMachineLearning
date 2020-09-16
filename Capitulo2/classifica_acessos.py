from dados import carregar_acessos

x, y = carregar_acessos()

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(x, y)

#teste = [[1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
resultado = modelo.predict(x)
diferencas = resultado - y
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(x)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(taxa_de_acerto)
print(total_de_elementos)
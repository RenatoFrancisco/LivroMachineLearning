import csv

def carregar_acessos():
    x = []
    y = []

    arquivo = open('acesso.csv', 'r')
    leitor = csv.reader(arquivo)

    for acessou_home, acessou_como_funciona, acessou_contato, comprou in leitor:
        x.append([acessou_home, acessou_como_funciona, acessou_contato])
        y.append(comprou)

    return x, y





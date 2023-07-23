from pickle import load
from obterDados import obterSimbolo
import numpy as np
import warnings

warnings.simplefilter(action='ignore')
classificador = load(open('modeloClassificadorMLP.pickle', 'rb'))

count = 0
actions = ['Nada', 'Compra', 'Venda']
while True:
    dados = obterSimbolo('WDO$', n=1000, delayCandles=count)
    hist = dados.copy().drop(columns=['spread'])
    x = np.append(hist.iloc[::-1].to_numpy().flatten(), dados.index[-1].hour)
    previsao = classificador.predict([x])
    
    if previsao[0] != 0:
        print(hist.index[-1], actions[previsao[0]])

    count += 1
from pickle import load
from obterDados import obterSimbolo
import numpy as np
import warnings

warnings.simplefilter(action='ignore')
classificador = load(open('modeloClassificadorMLP.pickle', 'rb'))

count = 0
actions = ['Nada', 'Compra', 'Venda']
lastPrev = 0
while True:
    dados = obterSimbolo('WDO$', n=1000, delayCandles=count)
    hist = dados.copy().drop(columns=['spread'])
    hist['hour'] = hist.index.hour
    histNP = hist.to_numpy()
    vmax = histNP[:, :4].max()
    vmin = histNP[:, :4].min()
    histNP[:, :4] = (histNP[:, :4] - vmin) / (vmax - vmin)
    vmax = histNP[:, 4].max()
    vmin = histNP[:, 4].min()
    histNP[:, 4] = (histNP[:, 4] - vmin) / (vmax - vmin)
    vmax = histNP[:, 5].max()
    vmin = histNP[:, 5].min()
    histNP[:, 5] = (histNP[:, 5] - vmin) / (vmax - vmin)
    histNP[:, 6] /= 24
    previsao = classificador.predict([histNP.flatten()])
    
    if lastPrev != int(previsao[0]):
        print('------------------------------------------')
    if int(previsao[0]) != 0:
        print(hist.index[-1], actions[int(previsao[0])])
    lastPrev = int(previsao[0])

    count += 1
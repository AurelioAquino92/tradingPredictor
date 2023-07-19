import pandas as pd
import numpy as np
from time import sleep, time
from pickle import load
from obterDados import obterSimbolo
import mplfinance as mplf
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
classificador = load(open('modeloClassificador.pickle', 'rb'))
colunas0 = ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume']
colunasY = []
for i in range(24):
    for name in colunas0:
        nomeCol = f'{name}-{str(i+1)}'
        colunasY.append(nomeCol)

while True:
    dados = obterSimbolo('WDO$', n=1001, delayCandles=0)
    hist = dados.copy().drop(columns=['spread'])
    print(hist.to_numpy().flatten())
    start = time()
    arraysNumpy = {}
    for coluna in colunas0:
        arraysNumpy[coluna] = hist[coluna].to_numpy(dtype=np.float32)
    histNP = np.zeros(len(arraysNumpy[coluna]))
    for i in range(1, 1000):
        for coluna in colunas0:
            histNP = np.roll(arraysNumpy[coluna], i)
            histNP[:i] = np.nan
            hist[coluna+str(i)] = histNP.tolist()
    hist = hist.dropna()
    previsao = classificador.predict(hist.to_numpy())
    print('Tempo: ', time()-start)
    print(hist)
    print(previsao)
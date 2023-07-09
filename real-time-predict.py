import pandas as pd
from time import sleep
from pickle import load
from obterDados import obterSimbolo
import mplfinance as mplf
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
regressor = load(open('modeloRegressor.pickle', 'rb'))
colunas0 = ['open', 'high', 'low', 'close']
colunasY = []
for i in range(24):
    for name in colunas0:
        nomeCol = f'{name}-{str(i+1)}'
        colunasY.append(nomeCol)

while True:
    real = obterSimbolo('WDO$', n=24, delayCandles=60)
    dados = obterSimbolo('WDO$', n=1001, delayCandles=24+50)
    hist = dados.copy().drop(columns=['tick_volume', 'real_volume', 'spread'])
    for i in range(1000):
        hist['open'+str(i+1)] = hist['open'].shift(i+1)
        hist['high'+str(i+1)] = hist['high'].shift(i+1)
        hist['low'+str(i+1)] = hist['low'].shift(i+1)
        hist['close'+str(i+1)] = hist['close'].shift(i+1)
    hist = hist.dropna()
    maxVs = hist.max(axis='columns')
    minVs = hist.min(axis='columns')
    histNorm = hist.subtract(minVs, axis=0).divide(maxVs - minVs, axis=0)
    previsaoNorm = pd.DataFrame(regressor.predict(histNorm), index=histNorm.index, columns=colunasY)
    previsao = previsaoNorm.multiply(maxVs - minVs, axis=0).add(minVs, axis=0)
    print(hist)
    print(previsao)
    previsaoTable = pd.DataFrame()
    for i in range(24):
        timestamp = previsao.iloc[[-1]].index + pd.Timedelta(minutes=5*(i+1))
        if timestamp.hour >= 18:
            timestamp += pd.Timedelta(days=1, hours=-9)
        previsaoX = {}
        for col in colunas0:
            previsaoX[col] = previsao.iloc[-1][col+'-'+str(i+1)]
            # previsaoTable = previsaoTable.drop(columns=col+'-'+str(i+1))
        previsaoTable = pd.concat([previsaoTable, pd.DataFrame([previsaoX], index=timestamp)])
    print(previsaoTable)
    fig = mplf.figure(figsize=(15, 4))
    mplf.plot(real, type='candle', ax=fig.add_subplot(2, 1, 1))
    mplf.plot(previsaoTable, type='candle', ax=fig.add_subplot(2, 1, 2))
    mplf.show()
    sleep(1)
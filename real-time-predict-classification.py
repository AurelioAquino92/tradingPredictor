from pickle import load
from obterDados import obterSimbolo
import numpy as np
from time import sleep
import warnings
from firebase_admin import credentials, firestore, initialize_app

classificador = load(open('models/modeloClassificadorXGBoost.pickle', 'rb'))
initialize_app(credentials.Certificate('key.json'))
db = firestore.client()
predictionsCollection = db.collection('predictions')

count = 0
actions = ['Nada', 'Compra', 'Venda']

lastPred = None
lastTimestamp = None

while True:
    dados = obterSimbolo('WDO$', n=1000, delayCandles=0)
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
    previsao = int(classificador.predict([histNP.flatten()])[0])
    
    timestamp = hist.index[-1]
    print(timestamp, actions[previsao])

    if previsao != 0:
        if lastTimestamp != hist.index[-1] or lastPred != previsao:
            print('escreve')

    lastPred = previsao
    lastTimestamp = hist.index[-1]
    sleep(10)
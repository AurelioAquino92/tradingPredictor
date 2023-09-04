from pickle import load
from obterDados import obterSimbolo
from time import sleep
from firebase_admin import credentials, firestore, initialize_app
from keras.models import load_model
import numpy as np
from datetime import datetime

initialize_app(credentials.Certificate('key.json'))
db = firestore.client()
predictionsCollection = db.collection('predictions')

modelos = [
    "ExtraTrees10000",
    "XGBoost10000"
]

classificadores = []
for modelo in modelos:
    classificadores.append(
        load(open(f'models/modeloClassificador{modelo}.pickle', 'rb'))
    )

modeloCNN = load_model('models/tf-cnn-model')

actionNames = ['Nada', 'Compra', 'Venda']

lastPred = -1
lastPrice = 0
lastTimestamp = None

def setPredictions(delay=0):
    global lastPred, lastPrice, lastTimestamp
    try:
        dados = obterSimbolo('WDO$N', n=1000, delayCandles=delay)
        hist = dados.copy().drop(columns=['spread'])
        hist['minute'] = hist.index.minute
        hist['hour'] = hist.index.hour
        hist['day_of_week'] = hist.index.day_of_week
        hist['day'] = hist.index.day
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
        histNP[:, 6] /= 60
        histNP[:, 7] /= 24
        histNP[:, 8] /= 4
        histNP[:, 9] /= 31

        previsoes = {}
        
        for classificador in classificadores:
            previsao = actionNames[int(classificador.predict([histNP.flatten()])[0])]
            previsoes[modelos[classificadores.index(classificador)]] = previsao
        previsoes['CNN'] = actionNames[np.argmax(modeloCNN.predict(np.array([histNP]), verbose=0)[0])]
        
        timestamp = hist.index[-1]
        if lastTimestamp != hist.index[-1] or lastPred != previsao:
            print('novas previsões:', timestamp, previsoes)
            file = open('lastTimeStamp.txt', 'w')
            file.write(str(hist.index[-1]))
            file.close()
        if lastPrice != hist['close'][-1]:
            predictionsCollection.document(str(timestamp)).set({
                'previsoes': previsoes,
                'price': hist['close'][-1]
            })

        lastPred = previsao
        lastPrice = hist['close'][-1]
        lastTimestamp = hist.index[-1]
    except Exception as e:
        print('Erro: ', e)

updateFrom = 0
try:
    file = open('lastTimeStamp.txt', 'r')
    lastTimestamp = datetime.strptime(file.read(), '%Y-%m-%d %H:%M:%S')
    while True:
        dados = obterSimbolo(delayCandles=updateFrom)
        if dados.index[-1] >= lastTimestamp:
            updateFrom += 1
        else:
            break
except Exception as e:
    print(e)

for i in range(updateFrom, -1, -1):
    setPredictions(i)

while True:
    setPredictions(i)
    sleep(5)
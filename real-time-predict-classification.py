from pickle import load
from obterDados import obterSimboloPosicao, obterSimboloData
from time import sleep
from firebase_admin import credentials, firestore, initialize_app
from keras.models import load_model
import numpy as np
from datetime import datetime

print('Iniciando conexão com Firebase...')
initialize_app(credentials.Certificate('key.json'))
db = firestore.client()
predictionsCollection = db.collection('predictions')

modelos = [
    "ExtraTrees",
    "XGBoost",
    "KNN"
]

print('Carregando modelos...')
classificadores = []
for modelo in modelos:
    classificadores.append(
        load(open(f'models/modeloClassificador{modelo}3006.pickle', 'rb'))
    )

modeloCNN = load_model('models/tf-cnn-model')
print('Finalizado! Iniciando previsões...')

actionNames = ['Nada', 'Compra', 'Venda']

lastPred = -1
lastPrice = 0
lastTimestamp = None

def setPredictions(delay=0):
    global lastPred, lastPrice, lastTimestamp
    try:
        histM5 = obterSimboloPosicao('WDO$N', n=300, delayCandles=i)
        histM5['minute'] = histM5.index.minute
        histM5['hour'] = histM5.index.hour
        histM5['day_of_week'] = histM5.index.day_of_week
        histM5['day'] = histM5.index.day
        histM5NP = histM5.to_numpy()
        hist = np.zeros((501, 6))
        hist[0, -4:] = histM5NP[-1][-4:]
        hist[1:301] = histM5NP[:, :6]
        histD1 = obterSimboloData(histM5.index[-1])
        histD1NP = histD1.to_numpy()
        hist[301:] = histD1NP
        vmax = hist[1:, :4].max()
        vmin = hist[1:, :4].min()
        hist[1:, :4] = (hist[1:, :4] - vmin) / (vmax - vmin)
        vmax = hist[1:, 4].max()
        vmin = hist[1:, 4].min()
        hist[1:, 4] = (hist[1:, 4] - vmin) / (vmax - vmin)
        vmax = hist[1:, 5].max()
        vmin = hist[1:, 5].min()
        hist[1:, 5] = (hist[1:, 5] - vmin) / (vmax - vmin)
        hist[0, 2] /= 60
        hist[0, 3] /= 24
        hist[0, 4] /= 4
        hist[0, 5] /= 31

        previsoes = {}
        
        for classificador in classificadores:
            previsao = actionNames[int(classificador.predict([hist.flatten()])[0])]
            previsoes[modelos[classificadores.index(classificador)]] = previsao
        stats = modeloCNN.predict(np.array([hist]), verbose=0)[0]
        previsoes['CNN'] = actionNames[np.argmax(stats)]
        
        timestamp = histM5.index[-1]
        if lastTimestamp != histM5.index[-1] or lastPred != previsao:
            print('novas previsões:', timestamp, previsoes)
            file = open('lastTimeStamp.txt', 'w')
            file.write(str(histM5.index[-1]))
            file.close()
        if lastPrice != histM5['close'][-1] or lastTimestamp != histM5.index[-1]:
            print('CNN Stats: ', np.round(stats, 3))
            predictionsCollection.document(str(timestamp)).set({
                'previsoes': previsoes,
                'price': histM5['close'][-1]
            })

        lastPred = previsao
        lastPrice = histM5['close'][-1]
        lastTimestamp = histM5.index[-1]
    except Exception as e:
        print('Erro: ', e)

updateFrom = 0
try:
    file = open('lastTimeStamp.txt', 'r')
    lastTimestamp = datetime.strptime(file.read(), '%Y-%m-%d %H:%M:%S')
    while True:
        dados = obterSimboloPosicao(delayCandles=updateFrom)
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
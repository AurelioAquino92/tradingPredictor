from pickle import load
from obterDados import obterSimbolo
from time import sleep
from firebase_admin import credentials, firestore, initialize_app
from keras.models import load_model
import numpy as np

initialize_app(credentials.Certificate('key.json'))
db = firestore.client()
predictionsCollection = db.collection('predictions')

modelos = [
    "ExtraTrees10000",
    # "KNN10000",
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

# for i in range(108*10, -1, -1):
while True:
    try:
        dados = obterSimbolo('WDO$N', n=1000, delayCandles=0)
        # dados = obterSimbolo('WDO$N', n=1000, delayCandles=i)
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
    sleep(5)
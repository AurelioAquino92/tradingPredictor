from pickle import load
from obterDados import obterSimbolo
from time import sleep
from firebase_admin import credentials, firestore, initialize_app

classificador = load(open('models/modeloClassificadorExtraTrees.pickle', 'rb'))
initialize_app(credentials.Certificate('key.json'))
db = firestore.client()
predictionsCollection = db.collection('predictions')

count = 0
actions = ['Nada', 'Compra', 'Venda']

lastPred = -1
lastTimestamp = None

while True:
    dados = obterSimbolo('WDO$N', n=1000, delayCandles=0)
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

    if lastTimestamp != hist.index[-1] or lastPred != previsao:
        print('nova previsão:', timestamp, actions[previsao])
        predictionsCollection.document(str(timestamp)).set({
            'previsao': actions[previsao],
            'price': hist['close'][-1]
        })

    lastPred = previsao
    lastTimestamp = hist.index[-1]
    sleep(5)
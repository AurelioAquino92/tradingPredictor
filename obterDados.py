import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def obterSimboloPosicao(simbolo='WDO$N', timeframe=mt5.TIMEFRAME_M5, delayCandles=0, n=99999):
    if not mt5.initialize():
        mt5.shutdown()
        return None
    dados = mt5.copy_rates_from_pos(simbolo, timeframe, delayCandles, n)
    if dados is None:
        return None
    dados = pd.DataFrame(dados)
    dados['time'] = pd.to_datetime(dados['time'], unit='s')
    dados.set_index('time', inplace=True)
    return dados.drop(columns=['spread'])

def obterSimboloData(dia, simbolo='WDO$N', timeframe=mt5.TIMEFRAME_D1, n=200):
    if not mt5.initialize():
        mt5.shutdown()
        return None
    dados = mt5.copy_rates_from(simbolo, timeframe, dia, n)
    if dados is None:
        return None
    dados = pd.DataFrame(dados)
    dados['time'] = pd.to_datetime(dados['time'], unit='s')
    dados.set_index('time', inplace=True)
    return dados.drop(columns=['spread'])
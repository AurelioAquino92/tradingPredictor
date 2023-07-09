import MetaTrader5 as mt5
import pandas as pd

def obterSimbolo(simbolo, timeframe=mt5.TIMEFRAME_M5, diaAtual=0, n=99999):
    if not mt5.initialize():
        mt5.shutdown()
        return None
    dados = mt5.copy_rates_from_pos(simbolo, timeframe, diaAtual, n)
    if dados is None:
        return None
    dados = pd.DataFrame(dados)
    dados['time'] = pd.to_datetime(dados['time'], unit='s')
    dados.set_index('time', inplace=True)
    return dados
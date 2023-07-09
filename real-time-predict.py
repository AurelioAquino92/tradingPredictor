import pandas as pd
from time import sleep
from obterDados import obterSimbolo

while True:
    dados = obterSimbolo('WDO$')
    sleep(1)
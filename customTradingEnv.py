import gym
import numpy as np
import pandas as pd

class CustomTradingEnv(gym.Env):
    def __init__(self, df_5min, df_daily, initial_balance=100000, take_profit=30, stop_loss=10,
                 observation_window_5min=300, observation_window_daily=200):
        super(CustomTradingEnv, self).__init__()
        self.df_5min = df_5min
        self.df_5min_closes = df_5min['close'].to_numpy()
        self.df_5min_highs = df_5min['high'].to_numpy()
        self.df_5min_lows = df_5min['low'].to_numpy()
        self.df_5min_volumes = df_5min['real_volume'].to_numpy()
        self.df_daily = df_daily
        self.df_daily_closes = df_daily['close'].to_numpy()
        self.df_daily_volumes = df_daily['real_volume'].to_numpy()
        self.initial_balance = initial_balance
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.observation_window_5min = observation_window_5min
        self.observation_window_daily = observation_window_daily
        self.reset()

        # Definir o espaço de ação e observação
        self.action_space = gym.spaces.Discrete(4)  # Três ações possíveis: fazer nada, comprar, vender, fechar operação
        self.observation_space = gym.spaces.Dict(
            {
                'M5_closes': gym.spaces.Box(low=0, high=1, shape=(self.observation_window_5min,), dtype=np.float32),
                'D1_closes': gym.spaces.Box(low=0, high=1, shape=(self.observation_window_daily,), dtype=np.float32),
                'M5_volumes': gym.spaces.Box(low=0, high=1, shape=(self.observation_window_5min,), dtype=np.float32),
                'D1_volumes': gym.spaces.Box(low=0, high=1, shape=(self.observation_window_daily,), dtype=np.float32),
                'shares_held': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                'price': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                'volume': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                'day': gym.spaces.Discrete(32, start=1),
                'dayofweek': gym.spaces.Discrete(5),
                'hour': gym.spaces.Discrete(19, start=9),
                'minute': gym.spaces.Discrete(60)
            }
        )

        # Definir a hora de fechamento das posições abertas
        self.closing_time = pd.Timestamp("17:00").time()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.observation_window_5min
        self.shares_held = 0
        self.profit = 0
        self.trades = []  # Manter um registro de todas as negociações
        self.done = False
        self.observation_5min = []
        self.observation_5min_volume = []
        self.observation_daily = []
        self.observation_daily_volume = []
        self.lot_size = 1
        return self._get_observation()

    def step(self, action):
        if self.done:
            raise ValueError("O episódio terminou. Chame reset() para reiniciar o ambiente.")

        self.current_step += 1

        if self.current_step >= len(self.df_5min):
            self.done = True
            return self._get_observation(), 0, self.done, {}

        current_price_5min = self.df_5min_closes[self.current_step]
        current_time = self.df_5min.index[self.current_step]

        if action == 1:  # Comprar
            cost = self.lot_size * current_price_5min
            self.balance -= cost
            self.shares_held += self.lot_size
            self.trades.append((self.current_step, "Buy", current_price_5min, self.shares_held))

        elif action == 2:  # Vender
            cost = -self.lot_size * current_price_5min
            self.balance -= cost
            self.shares_held += -self.lot_size
            self.trades.append((self.current_step, "Sell", current_price_5min, self.shares_held))

        elif action == 3 or (self.shares_held != 0 and current_time.time() >= self.closing_time): # Fechar posições abertas
            revenue = self.shares_held * current_price_5min
            self.balance += revenue
            self.shares_held = 0
            self.trades.append((self.current_step, "Close", current_price_5min, self.shares_held))

        # Testando um ambiente sem regras fixas de TP e SL
        # unrealized_profit = (current_price_5min - self.trades[-1][2]) * self.trades[-1][3] if self.trades else 0
        # reward = unrealized_profit       

        # reward
        if self.shares_held == 0:
            self.profit = self.balance - self.initial_balance
        reward = self.profit

        # Verificar se o lucro está abaixo de -1000 quando não há posições abertas
        # Ou se o passo atual está no final do dataframe
        if reward < -1000 or self.current_step == len(self.df_5min) - 1:
            self.done = True

        info = {
            'shares': self.shares_held,
            'profit (reward)': self.profit,
            'balance': self.balance,
            'step': self.current_step
        }
        if self.done:   # Adiciona trades apenas quando finalizar
            info['trades'] = self.trades

        return self._get_observation(), reward, self.done, info

    def _get_observation(self):
        # Obter preços do M5 e D1 para observação
        self.observation_5min = self.df_5min_closes[self.current_step-self.observation_window_5min:self.current_step]
        self.observation_5min_volume = self.df_5min_volumes[self.current_step-self.observation_window_5min:self.current_step]
        dailyStep = self.df_daily.index.get_loc(pd.Timestamp(self.df_5min.index[self.current_step].date()))
        self.observation_daily = self.df_daily_closes[dailyStep-self.observation_window_daily:dailyStep]
        self.observation_daily_volume = self.df_daily_volumes[dailyStep-self.observation_window_daily:dailyStep]

        # Normalização
        minVal = np.min(self.observation_5min)
        maxVal = np.max(self.observation_5min)
        self.observation_5min = (self.observation_5min - minVal) / (maxVal - minVal)

        minVal = np.min(self.observation_daily)
        maxVal = np.max(self.observation_daily)
        self.observation_daily = (self.observation_daily - minVal) / (maxVal - minVal)

        minVal = np.min(self.observation_5min_volume)
        maxVal = np.max(self.observation_5min_volume)
        self.observation_5min_volume = (self.observation_5min_volume - minVal) / (maxVal - minVal)

        minVal = np.min(self.observation_daily_volume)
        maxVal = np.max(self.observation_daily_volume)
        self.observation_daily_volume = (self.observation_daily_volume - minVal) / (maxVal - minVal)

        # TODO: adicionar dados de observações dos preços máximos e mínimos (Talvez não precise...)

        # Adicionar o dia da semana, dia do mês, hora e minuto do timestep atual
        current_time = self.df_5min.index[self.current_step]
        observation = {
            'M5_closes': self.observation_5min,
            'D1_closes': self.observation_daily,
            'M5_volumes': self.observation_5min_volume,
            'D1_volumes': self.observation_daily_volume,
            'shares_held': np.array([self.shares_held]),
            'price': np.array([self.df_5min_closes[self.current_step]]),
            'volume': np.array([self.df_5min_volumes[self.current_step]], dtype=np.int64),
            'day': current_time.day,
            'dayofweek': current_time.dayofweek,
            'hour': current_time.hour,
            'minute': current_time.minute,
        }
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

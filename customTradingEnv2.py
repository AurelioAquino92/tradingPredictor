import gym
import numpy as np
import pandas as pd

class CustomTradingEnv(gym.Env):
    def __init__(self, df_5min, df_daily):
        super(CustomTradingEnv, self).__init__()
        self.df_5min = df_5min
        self.df_5min_closes = df_5min['Close'].to_numpy()
        self.df_5min_highs = df_5min['High'].to_numpy()
        self.df_5min_lows = df_5min['Low'].to_numpy()
        self.df_5min_volumes = df_5min['Volume'].to_numpy()
        self.df_daily = df_daily
        self.df_daily_closes = df_daily['Close'].to_numpy()
        self.df_daily_volumes = df_daily['Volume'].to_numpy()
        self.observation_window_5min = 300
        self.observation_window_daily = 200
        self.reset()

        # Definir o espaço de ação e observação
        self.action_space = gym.spaces.Dict(
            {
                'operation': gym.spaces.Discrete(3),
                'take_profit': gym.spaces.Discrete(51),
                'stop_loss': gym.spaces.Discrete(51),
                'candle_limit': gym.spaces.Discrete(37, start=6)
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                'M5_closes': gym.spaces.Box(low=0, high=1, shape=(self.observation_window_5min,), dtype=np.float32),
                'M5_highs': gym.spaces.Box(low=0, high=1, shape=(self.observation_window_5min,), dtype=np.float32),
                'M5_lows': gym.spaces.Box(low=0, high=1, shape=(self.observation_window_5min,), dtype=np.float32),
                'M5_volumes': gym.spaces.Box(low=0, high=1, shape=(self.observation_window_5min,), dtype=np.float32),
                'D1_closes': gym.spaces.Box(low=0, high=1, shape=(self.observation_window_daily,), dtype=np.float32),
                'D1_volumes': gym.spaces.Box(low=0, high=1, shape=(self.observation_window_daily,), dtype=np.float32),
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
        self.current_step = self.observation_window_5min
        self.profit = 0
        self.trades = []  # Manter um registro de todas as negociações
        self.done = False
        self.observation_5min = []
        self.observation_5min_highs = []
        self.observation_5min_lows = []
        self.observation_5min_volume = []
        self.observation_daily = []
        self.observation_daily_volume = []
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

        operation_profit = None
        
        if action == 1:  # Comprar
            self.trades.append((self.current_step, "Buy", current_price_5min))
            for i in range(1, action['candle_limit']):
                if self.df_5min_highs[self.current_step + i] >= current_price_5min + action['take_profit']:
                    operation_profit = action['take_profit']
                    break
                elif self.df_5min_lows[self.current_step + i] <= current_price_5min - action['stop_loss']:
                    operation_profit = -action['stop_loss']
                    break
                # TODO: fechar operação às 17hs?
                # if (current_time.time() >= self.closing_time
                #     or self.current_step + i >= len(self.df_5min_closes)):
                #     operation_profit = 
            if operation_profit is None:
                operation_profit = self.df_5min_closes[self.current_step + action['candle_limit']] - current_price_5min

        elif action == 2:  # Vender
            self.trades.append((self.current_step, "Sell", current_price_5min))
            for i in range(1, action['candle_limit']):
                if self.df_5min_lows[self.current_step + i] <= current_price_5min - action['take_profit']:
                    operation_profit = action['take_profit']
                    break
                elif self.df_5min_highs[self.current_step + i] >= current_price_5min + action['stop_loss']:
                    operation_profit = -action['stop_loss']
                    break
                # TODO: fechar operação às 17hs?
                # if (current_time.time() >= self.closing_time
                #     or self.current_step + i >= len(self.df_5min_closes)):
                #     operation_profit = 
            if operation_profit is None:
                operation_profit = current_price_5min - self.df_5min_closes[self.current_step + action['candle_limit']]

        if operation_profit is None:
            reward = 0
        else:
            reward = operation_profit

        self.profit += reward

        # Verificar se o lucro está abaixo de -1000 quando não há posições abertas
        # Ou se o passo atual está no final do dataframe
        # O '40' dá a margem de segurança para ver o resultado da operação mais à frente
        if self.profit < -1000 or self.current_step > len(self.df_5min) - 40:
            self.done = True

        info = {
            'profit': self.profit,
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
        minVal = np.min([self.observation_5min, self.observation_5min_highs, self.observation_5min_lows])
        maxVal = np.max([self.observation_5min, self.observation_5min_highs, self.observation_5min_lows])
        
        self.observation_5min = (self.observation_5min - minVal) / (maxVal - minVal)
        self.observation_5min_highs = (self.observation_5min_highs - minVal) / (maxVal - minVal)
        self.observation_5min_lows = (self.observation_5min_lows - minVal) / (maxVal - minVal)

        minVal = np.min(self.observation_5min_volume)
        maxVal = np.max(self.observation_5min_volume)
        self.observation_5min_volume = (self.observation_5min_volume - minVal) / (maxVal - minVal)

        minVal = np.min(self.observation_daily)
        maxVal = np.max(self.observation_daily)
        self.observation_daily = (self.observation_daily - minVal) / (maxVal - minVal)

        minVal = np.min(self.observation_daily_volume)
        maxVal = np.max(self.observation_daily_volume)
        self.observation_daily_volume = (self.observation_daily_volume - minVal) / (maxVal - minVal)

        # TODO: adicionar dados de observações dos preços máximos e mínimos (Talvez não precise...)

        # Adicionar o dia da semana, dia do mês, hora e minuto do timestep atual
        current_time = self.df_5min.index[self.current_step]
        observation = {
            'M5_closes': self.observation_5min,
            'M5_highs': self.observation_5min_highs,
            'M5_lows': self.observation_5min_lows,
            'M5_volumes': self.observation_5min_volume,
            'D1_closes': self.observation_daily,
            'D1_volumes': self.observation_daily_volume,
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

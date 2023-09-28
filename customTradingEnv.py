import gym
import numpy as np
import pandas as pd

class CustomTradingEnv(gym.Env):
    def __init__(self, df_5min, df_daily, initial_balance=10000, take_profit=30, stop_loss=10,
                 observation_window_5min=300, observation_window_daily=200):
        super(CustomTradingEnv, self).__init__()
        self.df_5min = df_5min
        self.df_daily = df_daily
        self.initial_balance = initial_balance
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.observation_window_5min = observation_window_5min
        self.observation_window_daily = observation_window_daily
        self.reset()

        # Definir o espaço de ação e observação
        self.action_space = gym.spaces.Discrete(3)  # Três ações possíveis: fazer nada, comprar, vender
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.observation_window_5min + self.observation_window_daily + 7,), dtype=np.float32)

        # Definir a hora de fechamento das posições abertas
        self.closing_time = pd.Timestamp("17:00").time()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.shares_held = 0
        self.net_worth = self.balance
        self.trades = []  # Manter um registro de todas as negociações
        self.done = False
        self.observation_5min = []
        self.observation_daily = []
        self.open_position_time = None  # Hora de abertura da posição
        return self._get_observation()

    def step(self, action):
        if self.done:
            raise ValueError("O episódio terminou. Chame reset() para reiniciar o ambiente.")

        self.current_step += 1

        if self.current_step >= len(self.df_5min):
            self.done = True
            return self._get_observation(), 0, self.done, {}

        current_price_5min = self.df_5min.iloc[self.current_step]['Close']

        self.observation_5min = self.df_5min['Close'].to_numpy()[-self.observation_window_5min:].tolist()
        dailyStep = self.df_daily.index.get_loc(pd.Timestamp(self.df_5min.index[self.current_step].date()))
        self.observation_daily = self.df_daily['Close'].to_numpy()[dailyStep-self.observation_window_daily:dailyStep].tolist()

        if action == 1:  # Comprar
            if self.balance > 0:
                shares_to_buy = min(self.balance // current_price_5min, int(self.balance / 2))  # Limitar o número de ações compradas
                cost = shares_to_buy * current_price_5min
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.trades.append((self.current_step, "Buy", current_price_5min, shares_to_buy))
                self.open_position_time = self.df_5min.index[self.current_step]  # Registrar o timestamp de abertura da posição

        elif action == 2:  # Vender
            if self.shares_held > 0:
                revenue = self.shares_held * current_price_5min
                self.balance += revenue
                self.shares_held = 0
                self.trades.append((self.current_step, "Sell", current_price_5min, self.shares_held))
                self.open_position_time = None  # Zerar o timestamp de abertura da posição

        # Verificar se atingimos o take profit ou stop loss
        unrealized_profit = (current_price_5min - self.trades[-1][2]) * self.trades[-1][3] if self.trades else 0
        reward = unrealized_profit

        if self.open_position_time is not None:
            current_time = self.df_5min.index[self.current_step]
            if current_time.time() >= self.closing_time:
                # Fechar a posição automaticamente se estiver aberta após as 17:00
                self.done = True

        self.net_worth = self.balance + self.shares_held * current_price_5min

        # Verificar se o saldo está abaixo de -1000
        if self.net_worth < -1000:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        observation = np.array(self.observation_5min + self.observation_daily + [self.balance, self.shares_held, self.net_worth])
        # Adicionar o dia da semana, dia do mês, hora e minuto do timestep atual
        current_time = self.df_5min.index[self.current_step]
        day_of_week = current_time.dayofweek
        day_of_month = current_time.day
        current_hour = current_time.hour
        current_minute = current_time.minute
        observation = np.append(observation, [day_of_week, day_of_month, current_hour, current_minute])
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

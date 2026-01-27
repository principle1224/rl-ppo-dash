import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
import torch.nn as nn

SEED = 42
np.random.seed(SEED)


def fetch_data(start, end):
    """데이터 수집"""
    tickers = ["SPY", "TLT", "GLD", "DBC", "SHY"]
    df = pd.DataFrame()

    for ticker in tickers:
        df_temp = yf.download(ticker, start=start, end=end, progress=False)["Close"]
        df = pd.concat([df, df_temp], axis=1)
        time.sleep(1)

    df = df.dropna()
    df = df.pct_change().dropna()
    return df


class PortfolioEnvMonthly(gym.Env):
    """강화학습 환경"""
    
    def __init__(self, df, window_size=30, transaction_cost=0.002, rebalance_interval=20):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.rebalance_interval = rebalance_interval
        self.asset_dim = df.shape[1]
        
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.asset_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, self.asset_dim), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = self.window_size
        self.portfolio_value = 1.0
        self.weights = np.array([1.0 / self.asset_dim] * self.asset_dim)
        self.portfolio_returns = []
        self.drawdowns = []
        return self._get_observation(), {}

    def _get_observation(self):
        return self.df.iloc[
            self.current_step - self.window_size : self.current_step
        ].values

    def step(self, action):
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action = np.ones_like(action) / len(action)
        else:
            action /= np.sum(action)

        prev_weights = self.weights.copy()
        self.weights = action

        end_step = min(self.current_step + self.rebalance_interval, len(self.df))
        segment = self.df.iloc[self.current_step:end_step].values

        peak_value = self.portfolio_value
        for daily_return in segment:
            portfolio_return = np.dot(self.weights, daily_return)
            self.portfolio_value *= (1 + portfolio_return)
            self.portfolio_returns.append(portfolio_return)

            peak_value = max(peak_value, self.portfolio_value)
            drawdown = (peak_value - self.portfolio_value) / peak_value
            self.drawdowns.append(drawdown)

        turnover = np.sum(np.abs(self.weights - prev_weights))
        cost = turnover * self.transaction_cost
        self.portfolio_value -= cost

        reward = self._calculate_reward()
        self.current_step = end_step
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, False, {
            "portfolio_value": self.portfolio_value
        }

    def _calculate_reward(self):
        λ1, λ2, λ3 = 3.0, 1.5, 0.5

        if len(self.portfolio_returns) >= self.rebalance_interval:
            window_returns = np.array(
                self.portfolio_returns[-self.rebalance_interval:]
            )
            sharpe = np.mean(window_returns) / (np.std(window_returns) + 1e-8)
        else:
            sharpe = 0

        max_dd = max(self.drawdowns[-self.rebalance_interval:] or [0])
        log_return = np.log(self.portfolio_value)
        transaction_penalty = np.sum(np.abs(self.weights))

        reward = (
            log_return - λ1 * max_dd + λ2 * sharpe - λ3 * transaction_penalty
        )
        return reward


def run_tuned_strategy():
    """Optuna 최적화 전략 실행"""
    
    # 학습
    print("[Optuna] 학습 데이터 수집 중...")
    train_df = fetch_data("2005-01-01", "2023-12-31")
    train_env = PortfolioEnvMonthly(train_df)
    
    print(" [Optuna] 모델 학습 중 (PDF 슬라이드 12 최적 파라미터)...")
    
    policy_kwargs = dict(
        activation_fn=nn.SiLU,
        net_arch=[dict(pi=[64, 32], vf=[64, 32])]
    )
    
    model = PPO(
        "MlpPolicy", 
        train_env, 
        policy_kwargs=policy_kwargs,
        clip_range=0.26035737930860936,
        gamma=0.9949518295000087,
        ent_coef=0.03964004687913023,
        learning_rate=0.0002397841523159722,
        verbose=0, 
        seed=SEED
    )
    
    model.learn(total_timesteps=300_000)
    
    print(" [Optuna] 모델 저장 중...")
    os.makedirs("models", exist_ok=True)
    model_path = "models/ray_dalio_tuned_model"
    model.save(model_path)
    
    # 테스트
    print(" [Optuna] 테스트 데이터 수집 중...")
    test_df = fetch_data("2024-01-01", "2025-12-31")  
    test_env = PortfolioEnvMonthly(test_df)
    
    print(" [Optuna] 전략 평가 중...")
    loaded_model = PPO.load(model_path, env=test_env)
    
    obs, _ = test_env.reset()
    portfolio_values = [test_env.portfolio_value]
    weights_history = []  
    done = False
    
    while not done:
        action, _ = loaded_model.predict(obs, deterministic=True)
        weights_history.append(action.copy())  
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        portfolio_values.append(info["portfolio_value"])
    
    # 성과 계산
    final_value = portfolio_values[-1]
    total_return = (final_value - 1.0) * 100
    
    portfolio_returns = np.array(test_env.portfolio_returns)
    sharpe = (np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8)) * np.sqrt(252)
    max_drawdown = max(test_env.drawdowns) if test_env.drawdowns else 0
    
    return {
        "portfolio_values": portfolio_values,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_value": final_value,
        "returns": portfolio_returns,        
        "drawdowns": test_env.drawdowns,     
        "weights": np.array(weights_history) 
    }
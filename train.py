import gym
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3 import PPO
from generate_scenario import generate_scenario
import os

# 사용자 정의 환경
class PortfolioEnv(gym.Env):
    def __init__(self, seed):
        super(PortfolioEnv, self).__init__()
        # 52-dim state (예: 10개 주식의 일단위 수익률, 63일 평균 수익률, 63일 표준편차, Relative Volume, VIX 지수, 5년 국채금리, previous action)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        # 10-dim action: 각 주식의 가중치 (0~1 사이 값, 이후 정규화하여 합=1)
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        
        # Load data for the current episode and remove the first column
        data = None
        while data is None:
            data = generate_scenario(10, seed)
        raw_data = data.iloc[:, 1:].values
        # Initialize self.data with 42 states from raw_data and append ten 0s for previous action
        self.data = np.hstack((raw_data, np.zeros((raw_data.shape[0], 10))))
        self.max_steps = len(self.data)  # Number of rows in the CSV file
        self.current_step = 0
        self.previous_action = np.zeros(10)  # Initialize previous action as zeros
        self.state = self._next_state()

    def _next_state(self):
        # Use the current row of the CSV file and append previous action as the state
        return np.hstack((self.data[self.current_step, :42], self.previous_action)).astype(np.float32)

    def reset(self):
        self.current_step = 0
        self.previous_action = np.zeros(10)  # Reset previous action to zeros
        self.state = self._next_state()
        return self.state

    def step(self, action):
        # action 정규화: 모든 가중치의 합이 1이 되도록 처리
        weights = action / (np.sum(action) + 1e-8)
        
        # 예시: state의 각 4개 열의 첫 번째 값이 각 주식의 일일 수익률라고 가정
        stock_returns = self.state[:40:4]
        portfolio_return = np.dot(weights, stock_returns)
        
        stock_vols = self.state[2:42:4]
        
        # 위험 패널티: 포트폴리오의 분산을 위험 지표로 사용 (계수는 하이퍼파라미터)
        portfolio_vol = np.sqrt(np.sum((weights * stock_vols) ** 2))
        
        # Calculate turnover as the sum of absolute differences between current and previous actions
        turnover = np.sum(np.abs(weights - self.previous_action))
        
        # Reward: 수익률에서 위험 패널티와 turnover 패널티를 차감 (목표: 수익은 높이고 위험과 turnover는 낮게)
        reward = portfolio_return - 0.1 * portfolio_vol - 0.01 * turnover

        # 다음 state 업데이트 (실제 환경에서는 시장 데이터, 기술적 지표 등으로 전이)
        self.current_step += 1
        if self.current_step < self.max_steps:
            self.state = self._next_state()
        
        # Update previous action
        self.previous_action = weights
        
        done = self.current_step >= self.max_steps
        info = {"portfolio_return": portfolio_return, "risk_penalty": portfolio_vol}
        return self.state, reward, done, info

    def render(self, mode="human"):
        # 간단한 출력 예시 (필요 시 구체적으로 구현)
        # print(f"Step: {self.current_step}")
        return 1

def main():
    model_path = "ppo_portfolio"  # 모델 파일 이름 (예: ppo_portfolio.zip)
    total_episodes = 10000         # 총 에피소드 수
    test_rollout_steps = 10        # 평가를 위한 테스트 스텝 수

    for episode_index in range(total_episodes):
        # 매 에피소드마다 고유 시드를 사용하여 환경 생성 (재현성을 위해)
        env = PortfolioEnv(seed=episode_index)
        
        # 저장된 모델이 있으면 불러오고, 없으면 새로 생성
        if os.path.exists(model_path + ".zip"):
            model = PPO.load(model_path, env=env)
            print(f"Episode {episode_index}: 모델 불러오기 완료 ({model_path})")
        else:
            model = PPO("MlpPolicy", env, verbose=1)
            print(f"Episode {episode_index}: 새 모델 생성")
        
        # 에피소드 내 학습: 한 에피소드당 max_steps만큼 학습
        model.learn(total_timesteps=env.max_steps)
        
        # 평가: 학습된 모델로 테스트 롤아웃 수행
        obs = env.reset()
        returns = []
        for i in range(test_rollout_steps):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            returns.append(info["portfolio_return"])
            env.render()
            if done:
                break
        
        # Sharpe Ratio 계산 (평균 수익률 / 표준편차)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / (std_return + 1e-8)  # 0으로 나누는 경우 방지
        print(f"Episode {episode_index} Sharpe Ratio: {sharpe_ratio:.4f}")
        
        # 매 10 에피소드마다 모델 저장
        if episode_index % 10 == 9:
            model.save(model_path)
            print("모델 학습 후 저장 완료:", model_path)

if __name__ == '__main__':
    main()
        
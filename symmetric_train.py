import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from generate_scenario import generate_scenario
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import datetime
import logging
import traceback
import threading
import copy
from app.server import start_server
from app.utils import setup_logger

# Get logger
logger = setup_logger()

# 커스텀 신경망 정책 (주식 순서 무관한 대칭 구조를 위한)
class SymmetricStockPolicy(nn.Module):
    """
    주식 순서에 무관한 대칭적 신경망 정책
    - 각 주식마다 동일한 가치 네트워크 사용
    - 모든 주식의 특성을 통합하여 최종 행동 결정
    """
    def __init__(self, observation_space, action_space, num_stocks=10, features_per_stock=4, window_size=10):
        super(SymmetricStockPolicy, self).__init__()
        
        self.num_stocks = num_stocks  # 주식 수
        self.features_per_stock = features_per_stock  # 주식당 특성 수
        self.window_size = window_size  # 시간 윈도우 크기
        
        # 각 주식의 입력 크기
        self.stock_input_dim = features_per_stock * window_size
        
        # 매크로 데이터 차원 (VIX, 국채 수익률)
        self.macro_dim = 2
        
        # 1. 개별 주식 가치 네트워크 (모든 주식이 공유)
        self.stock_value_net = nn.Sequential(
            nn.Linear(self.stock_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # 각 주식의 잠재 표현(latent representation)
        )
        
        # 2. 이전 행동 인코더
        self.action_encoder = nn.Sequential(
            nn.Linear(num_stocks, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # 3. 매크로 데이터 인코더
        self.macro_encoder = nn.Sequential(
            nn.Linear(self.macro_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # 4. 통합 네트워크 (모든 주식의 잠재 표현 + 매크로 + 이전 행동)
        self.integration_net = nn.Sequential(
            nn.Linear(16 * num_stocks + 16 + 16, 128),  # 주식 잠재표현 + 매크로 + 이전행동
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 5. 정책 헤드 (행동 출력)
        self.policy_head = nn.Linear(64, num_stocks)
        
        # 6. 가치 헤드 (상태 가치 출력)
        self.value_head = nn.Linear(64, 1)
        
        # 행동 분포의 표준편차
        self.log_std = nn.Parameter(torch.zeros(num_stocks))
    
    def forward(self, obs):
        """
        관측값을 받아 행동 분포와 상태 가치 반환
        """
        batch_size = obs.shape[0]
        
        # 관측값에서 각 주식의 데이터, 매크로 데이터, 이전 행동 분리
        stock_features = []
        for i in range(self.num_stocks):
            # i번째 주식의 특성 추출
            start_idx = i * self.features_per_stock * self.window_size
            end_idx = start_idx + self.features_per_stock * self.window_size
            stock_feature = obs[:, start_idx:end_idx]
            stock_features.append(stock_feature)
        
        # 매크로 데이터 추출
        macro_start = self.num_stocks * self.features_per_stock * self.window_size
        macro_data = obs[:, macro_start:macro_start + self.macro_dim]
        
        # 이전 행동 추출
        prev_action = obs[:, -self.num_stocks:]
        
        # 각 주식의 특성을 개별적으로 처리 (대칭적 처리)
        stock_embeddings = []
        for i in range(self.num_stocks):
            embedding = self.stock_value_net(stock_features[i])
            stock_embeddings.append(embedding)
        
        # 모든 주식 임베딩을 연결
        all_stock_embedding = torch.cat(stock_embeddings, dim=1)
        
        # 매크로 데이터 인코딩
        macro_embedding = self.macro_encoder(macro_data)
        
        # 이전 행동 인코딩
        action_embedding = self.action_encoder(prev_action)
        
        # 모든 특성 통합
        combined = torch.cat([all_stock_embedding, macro_embedding, action_embedding], dim=1)
        
        # 통합 네트워크 통과
        integrated = self.integration_net(combined)
        
        # 정책 출력 (행동 평균)
        action_mean = self.policy_head(integrated)
        
        # 행동 분포 생성
        action_std = torch.exp(self.log_std)
        
        # 가치 함수 출력
        value = self.value_head(integrated)
        
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        """
        관측값으로부터 행동 샘플링
        """
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            # 결정적 행동 (평균값)
            action = action_mean
        else:
            # 확률적 행동 (정규 분포에서 샘플링)
            normal = Normal(action_mean, action_std)
            action = normal.sample()
        
        return action, value
    
    def evaluate_actions(self, obs, actions):
        """
        행동 평가 (로그 확률, 엔트로피, 가치 계산)
        """
        action_mean, action_std, value = self.forward(obs)
        
        normal = Normal(action_mean, action_std)
        log_probs = normal.log_prob(actions).sum(dim=1, keepdim=True)
        entropy = normal.entropy().sum(dim=1, keepdim=True)
        
        return log_probs, entropy, value

# 대칭적 포트폴리오 환경 (주식 순서 무관)
class SymmetricPortfolioEnv(gym.Env):
    """
    주식 순서에 무관한 대칭적 포트폴리오 환경
    - 주식 순서가 바뀌어도 동일한 성능 보장
    """
    def __init__(self, seed, window_size=10):
        super(SymmetricPortfolioEnv, self).__init__()
        
        self.window_size = window_size
        self.num_stocks = 10  # 주식 수
        self.features_per_stock = 4  # 주식당 특성 수
        
        # 상태 공간: 10개 주식 * (4개 특성 * window_size) + 매크로 데이터(2개) + 이전 행동(10개)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.num_stocks * self.features_per_stock * window_size + 2 + self.num_stocks,),
            dtype=np.float32
        )
        
        # 행동 공간: 10개 주식에 대한 가중치 (-1에서 1 사이)
        self.action_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(self.num_stocks,), 
            dtype=np.float32
        )
        
        # 데이터 로드 및 초기화
        self.seed_value = seed
        self._load_data()
        
        self.max_steps = len(self.market_data) - window_size
        self.current_step = 0
        self.previous_action = np.zeros(self.num_stocks)
        
        # 포트폴리오 성과 추적
        self.portfolio_value = 1.0
        self.portfolio_values = [1.0]
        
        # 초기 상태 설정
        self.state = self._get_state()
    
    def _load_data(self):
        """
        시나리오 데이터 로드
        """
        data = None
        while data is None:
            data = generate_scenario(self.num_stocks, self.seed_value)
        
        # 주식 데이터와 매크로 데이터 분리
        self.market_data = data.iloc[:, 1:41].values  # 10개 주식 * 4개 특성 = 40
        self.macro_data = data.iloc[:, 41:43].values  # 매크로 데이터 (VIX, 국채 수익률)
        
        logger.info(f"Environment initialized with data shape: {self.market_data.shape}")
    
    def _get_state(self):
        """
        현재 상태(관측값) 반환
        """
        # 현재 윈도우의 주식 데이터
        market_window = self.market_data[self.current_step:self.current_step + self.window_size]
        
        # 현재 매크로 데이터
        current_macro = self.macro_data[self.current_step + self.window_size - 1]
        
        # 데이터 결합
        return np.concatenate([
            market_window.flatten(),  # 10개 주식 * 4개 특성 * window_size
            current_macro,           # 매크로 데이터 2개
            self.previous_action      # 이전 행동 10개
        ]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """
        환경 초기화
        """
        if seed is not None:
            self.seed_value = seed
            self._load_data()
        
        self.current_step = 0
        self.previous_action = np.zeros(self.num_stocks)
        self.portfolio_value = 1.0
        self.portfolio_values = [1.0]
        self.state = self._get_state()
        
        return self.state, {}
    
    def shuffle_stocks(self):
        """
        주식 순서를 무작위로 섞음 (대칭 특성 테스트용)
        """
        # 셔플 인덱스 생성
        shuffle_idx = np.random.permutation(self.num_stocks)
        
        # 시장 데이터 재배열
        new_market_data = np.zeros_like(self.market_data)
        
        for i, idx in enumerate(shuffle_idx):
            # 각 주식의 4개 특성 이동
            src_start = idx * 4
            src_end = src_start + 4
            
            dst_start = i * 4
            dst_end = dst_start + 4
            
            new_market_data[:, dst_start:dst_end] = self.market_data[:, src_start:src_end]
        
        # 데이터 업데이트
        self.market_data = new_market_data
        
        # 이전 행동도 재배열
        self.previous_action = self.previous_action[shuffle_idx]
        
        # 상태 업데이트
        self.state = self._get_state()
        
        return shuffle_idx
    
    def step(self, action):
        """
        환경에서 한 스텝 진행
        """
        # 이전 가중치 저장
        prev_weights = self.previous_action.copy()
        
        # 정규화: 평균이 0이고 절대값 합이 1이 되도록
        action = action - np.mean(action)
        weights = action / (np.sum(np.abs(action)) + 1e-8)
        
        # 현재 가중치 저장
        self.previous_action = weights.copy()
        
        # 다음 시간 단계로 이동
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 새 시점에서의 수익률 데이터 가져오기
        if not terminated:
            # 주식별 수익률 데이터 인덱스 (각 주식의 첫 번째 특성)
            returns_indices = np.arange(0, 40, 4)
            
            # 백분율에서 소수점으로 변환 (1% -> 0.01)
            stock_returns = self.market_data[self.current_step + self.window_size - 1, returns_indices] / 100.0
            
            # 주식별 변동성 데이터 인덱스 (각 주식의 세 번째 특성)
            vol_indices = np.arange(2, 40, 4)
            stock_vols = self.market_data[self.current_step + self.window_size - 1, vol_indices] / 100.0
            
            # 포트폴리오 수익률 계산 (가중치 * 수익률)
            portfolio_return = np.sum(weights * stock_returns)
            
            # 포트폴리오 변동성 계산
            portfolio_vol = np.sqrt(np.sum((weights * stock_vols) ** 2))
            
            # 거래 회전율 계산 (이전 가중치와 현재 가중치의 차이)
            turnover = np.sum(np.abs(weights - prev_weights))
            
            # 거래 비용
            transaction_cost = 0.001 * turnover
            
            # 순 수익률
            net_return = portfolio_return - transaction_cost
            
            # 포트폴리오 가치 업데이트
            self.portfolio_value *= (1 + net_return)
            self.portfolio_values.append(self.portfolio_value)
            
            # 보상 계산: 수익률 - 위험 패널티 - 거래 비용
            reward = portfolio_return - 0.5 * portfolio_vol - transaction_cost
            
            # 새 상태 계산
            self.state = self._get_state()
        else:
            # 에피소드 종료 시 기본값
            portfolio_return = 0
            portfolio_vol = 0
            turnover = 0
            net_return = 0
            reward = 0
        
        # 정보 딕셔너리
        info = {
            "portfolio_return": portfolio_return if not terminated else 0,
            "net_return": net_return if not terminated else 0,
            "portfolio_vol": portfolio_vol if not terminated else 0,
            "turnover": turnover if not terminated else 0,
            "portfolio_value": self.portfolio_value
        }
        
        return self.state, reward, terminated, truncated, info

# PPO 에이전트를 위한 대칭 정책
class SymmetricPPOPolicy(torch.nn.Module):
    """
    PPO에서 사용할 대칭적 정책 네트워크
    """
    def __init__(self, observation_space, action_space):
        super(SymmetricPPOPolicy, self).__init__()
        
        # 입력 및 출력 차원
        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        
        # 대칭 구조 설정
        self.num_stocks = 10
        self.features_per_stock = 4
        self.window_size = 10
        
        # 신경망 구성
        self.symmetric_net = SymmetricStockPolicy(
            observation_space, 
            action_space,
            num_stocks=self.num_stocks,
            features_per_stock=self.features_per_stock,
            window_size=self.window_size
        )
        
        # 디바이스 속성 추가 (나중에 설정됨)
        self.device = torch.device("cpu")
    
    def forward(self, obs):
        """
        관측값을 받아 행동 분포와 가치 반환
        """
        return self.symmetric_net(obs)
    
    def to(self, device):
        """디바이스 이동 시 디바이스 속성 업데이트"""
        self.device = device
        return super().to(device)

# 대칭 정책을 사용하는 PPO
class SymmetricPPO:
    """
    대칭 정책 네트워크를 사용하는 PPO 구현
    """
    def __init__(self, env, learning_rate=0.0003, n_steps=2048, batch_size=64,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, 
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, device="auto"):
        
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        # GPU 사용 설정
        self.device = device
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # 정책 네트워크 생성 및 GPU로 이동
        self.policy = SymmetricPPOPolicy(env.observation_space, env.action_space)
        self.policy = self.policy.to(self.device)
        
        # 옵티마이저
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 학습 버퍼
        self.buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': []
        }
        
        # 학습 정보
        self.n_updates = 0
        self.total_timesteps = 0
    
    def predict(self, observation, deterministic=False):
        """
        정책으로 행동 예측
        """
        with torch.no_grad():
            # GPU로 관측값 이동
            if isinstance(observation, np.ndarray):
                observation = torch.FloatTensor(observation).to(self.device)
                
            # 배치 차원 확인
            if observation.ndim == 1:
                observation = observation.unsqueeze(0)
                
            # 예측 실행
            action_mean, action_std, _ = self.policy.symmetric_net(observation)
            
            if deterministic:
                action = action_mean
            else:
                # 정규 분포에서 샘플링
                normal = torch.distributions.Normal(action_mean, action_std)
                action = normal.sample()
                
            # CPU로 결과 이동 및 NumPy 변환
            action = action.cpu().numpy()
            if action.shape[0] == 1:
                action = action[0]
                
            return action, None
    
    def learn(self, total_timesteps, callback=None):
        """
        PPO 알고리즘으로 학습
        """
        timesteps_so_far = 0
        
        while timesteps_so_far < total_timesteps:
            # 1. 데이터 수집
            self._collect_rollouts()
            timesteps_so_far += self.n_steps
            self.total_timesteps += self.n_steps
            
            # 2. 학습 준비
            obs, actions, returns, advantages = self._prepare_training_data()
            
            # 3. 미니배치로 학습
            self._update_policy(obs, actions, returns, advantages)
            
            # 콜백 호출
            if callback is not None:
                if not callback.on_step():
                    break
                    
            # 진행 상황 로깅
            if timesteps_so_far % 5000 == 0:
                logger.info(f"Timesteps: {timesteps_so_far}/{total_timesteps} - Updates: {self.n_updates}")
        
        return self
    
    def _collect_rollouts(self):
        """
        환경과 상호작용하여 롤아웃 데이터 수집
        """
        # 버퍼 초기화
        for key in self.buffer.keys():
            self.buffer[key] = []
        
        obs, _ = self.env.reset()
        done = False
        
        for _ in range(self.n_steps):
            # 행동 예측 및 가치 계산
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_mean, action_std, value = self.policy.symmetric_net(obs_tensor)
                normal = torch.distributions.Normal(action_mean, action_std)
                action = normal.sample()
                log_prob = normal.log_prob(action).sum(dim=1)
                
                action = action.cpu().numpy()[0]
                value = value.cpu().numpy()[0, 0]
                log_prob = log_prob.cpu().numpy()[0]
            
            # 환경에서 스텝 실행
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # 버퍼에 저장
            self.buffer['observations'].append(obs)
            self.buffer['actions'].append(action)
            self.buffer['rewards'].append(reward)
            self.buffer['dones'].append(done)
            self.buffer['values'].append(value)
            self.buffer['log_probs'].append(log_prob)
            
            # 다음 상태로 이동
            obs = next_obs
            
            # 에피소드 종료 시 환경 리셋
            if done or truncated:
                obs, _ = self.env.reset()
                done = False
    
    def _prepare_training_data(self):
        """
        학습 데이터 준비 (GAE 계산 등)
        """
        # 버퍼 데이터를 NumPy 배열로 변환
        observations = np.array(self.buffer['observations'], dtype=np.float32)
        actions = np.array(self.buffer['actions'], dtype=np.float32)
        rewards = np.array(self.buffer['rewards'], dtype=np.float32)
        dones = np.array(self.buffer['dones'], dtype=np.float32)
        values = np.array(self.buffer['values'], dtype=np.float32)
        
        # GAE 계산
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(self.n_steps - 1)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t]
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        
        # 텐서로 변환하고 GPU로 이동
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        
        # 어드밴티지 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return observations, actions, returns, advantages
    
    def _update_policy(self, observations, actions, returns, advantages):
        """
        수집된 데이터로 정책 업데이트
        """
        # TensorDataset과 DataLoader 생성
        dataset = torch.utils.data.TensorDataset(observations, actions, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 미니배치로 업데이트
        for _ in range(10):  # 에포크 수
            for batch_obs, batch_actions, batch_returns, batch_advantages in loader:
                # 현재 정책으로 행동 평가
                log_probs, entropy, values = self.policy.symmetric_net.evaluate_actions(batch_obs, batch_actions)
                
                # 이전 로그 확률 계산
                with torch.no_grad():
                    old_log_probs, _, _ = self.policy.symmetric_net.evaluate_actions(batch_obs, batch_actions)
                
                # 비율 계산 (π_new / π_old)
                ratio = torch.exp(log_probs - old_log_probs)
                
                # 클리핑된 서라운드 목적 함수
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 가치 함수 손실
                value_loss = F.mse_loss(values, batch_returns)
                
                # 엔트로피 손실
                entropy_loss = -entropy.mean()
                
                # 총 손실
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # 경사 하강
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        self.n_updates += 1

# 대칭성 테스트 환경 (주식 순서를 섞어도 동일한 성능 확인)
class SymmetryTestCallback:
    """
    학습 중 대칭성을 테스트하는 커스텀 콜백
    - 주기적으로 주식 순서를 섞고 성능 비교
    - StableBaselines의 BaseCallback과 달리 직접 모델 참조를 받음
    """
    def __init__(self, model, eval_env, eval_freq=10000, verbose=1):
        self.model = model  # 모델 직접 참조
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.verbose = verbose
        self.best_mean_reward = -np.inf
        self.n_calls = 0  # 호출 횟수 카운터
    
    def on_step(self):
        """콜백 단계 실행"""
        self.n_calls += 1
        
        # 주기적으로 대칭성 테스트
        if self.n_calls % self.eval_freq == 0:
            logger.info("\n===== Testing Symmetry =====")
            
            # 1. 원본 순서로 성능 평가
            mean_reward, std_reward = self._evaluate_policy()
            logger.info(f"Original order - Mean reward: {mean_reward:.5f} +/- {std_reward:.5f}")
            
            # 2. 순서를 섞고 성능 평가
            shuffle_idx = self.eval_env.shuffle_stocks()
            logger.info(f"Shuffled stocks with permutation: {shuffle_idx}")
            
            mean_reward_shuffled, std_reward_shuffled = self._evaluate_policy()
            logger.info(f"Shuffled order - Mean reward: {mean_reward_shuffled:.5f} +/- {std_reward_shuffled:.5f}")
            
            # 3. 대칭성 점수 계산 (점수 차이의 절대값)
            symmetry_score = abs(mean_reward - mean_reward_shuffled)
            logger.info(f"Symmetry score (lower is better): {symmetry_score:.5f}")
            
            if symmetry_score < 0.05:
                logger.info("Good symmetry: Performance is consistent regardless of stock order")
            else:
                logger.info("Poor symmetry: Performance varies based on stock order")
        
        return True
    
    def _evaluate_policy(self, n_eval_episodes=5):
        """
        정책 평가
        """
        all_rewards = []
        
        for _ in range(n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_rewards = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_rewards.append(reward)
                done = terminated or truncated
            
            all_rewards.append(sum(episode_rewards))
        
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        
        return mean_reward, std_reward

# 메인 함수
def main():
    try:
        logger.info("="*50)
        logger.info(f"Symmetric Portfolio Optimization Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # GPU 사용 가능 여부 확인
        if torch.cuda.is_available():
            logger.info(f"GPU is available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.info("GPU is not available, using CPU")
        logger.info("="*50)
        
        # 환경 생성
        env = SymmetricPortfolioEnv(seed=42, window_size=10)
        logger.info("Created symmetric portfolio environment")
        
        # PPO 모델 생성 - GPU 사용 설정
        model = SymmetricPPO(
            env,
            learning_rate=0.0001,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cuda"  # GPU 사용 명시
        )
        logger.info("Created symmetric PPO model with GPU acceleration")
        
        # 평가 환경 생성
        eval_env = SymmetricPortfolioEnv(seed=1234, window_size=10)
        
        # 대칭성 테스트 콜백 생성
        symmetry_callback = SymmetryTestCallback(
            model=model,  # 모델 직접 전달
            eval_env=eval_env,
            eval_freq=10000
        )
        
        # 학습 실행
        logger.info("Starting training...")
        model.learn(
            total_timesteps=500000,
            callback=symmetry_callback
        )
        logger.info("Training completed")
        
        # 최종 모델 저장
        torch.save(model.policy.state_dict(), "symmetric_portfolio_policy.pt")
        logger.info("Model saved as symmetric_portfolio_policy.pt")
        
        # 최종 평가
        logger.info("\n===== Final Evaluation =====")
        returns = []
        for i in range(10):
            # 다양한 시드로 테스트
            test_env = SymmetricPortfolioEnv(seed=5000 + i, window_size=10)
            obs, _ = test_env.reset()
            done = False
            portfolio_values = [1.0]
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                if not done:
                    portfolio_values.append(info["portfolio_value"])
            
            # 최종 수익률 계산
            final_return = (portfolio_values[-1] - 1.0) * 100
            returns.append(final_return)
            logger.info(f"Test {i+1} - Final return: {final_return:.2f}%")
        
        # 평균 수익률
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        logger.info(f"Average return: {avg_return:.2f}% +/- {std_return:.2f}%")
        
        # 대칭성 최종 테스트
        logger.info("\n===== Final Symmetry Test =====")
        
        # 원본 순서 테스트
        final_test_env = SymmetricPortfolioEnv(seed=9999, window_size=10)
        obs, _ = final_test_env.reset()
        done = False
        original_values = [1.0]
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = final_test_env.step(action)
            done = terminated or truncated
            if not done:
                original_values.append(info["portfolio_value"])
        
        original_return = (original_values[-1] - 1.0) * 100
        logger.info(f"Original order - Final return: {original_return:.2f}%")
        
        # 섞은 순서 테스트
        final_test_env = SymmetricPortfolioEnv(seed=9999, window_size=10)
        obs, _ = final_test_env.reset()
        # 주식 순서 섞기
        shuffle_idx = final_test_env.shuffle_stocks()
        logger.info(f"Shuffled with permutation: {shuffle_idx}")
        
        done = False
        shuffled_values = [1.0]
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = final_test_env.step(action)
            done = terminated or truncated
            if not done:
                shuffled_values.append(info["portfolio_value"])
        
        shuffled_return = (shuffled_values[-1] - 1.0) * 100
        logger.info(f"Shuffled order - Final return: {shuffled_return:.2f}%")
        
        # 대칭성 점수
        symmetry_score = abs(original_return - shuffled_return)
        logger.info(f"Final symmetry score: {symmetry_score:.2f}")
        
        if symmetry_score < 1.0:
            logger.info("Excellent symmetry: Model is invariant to stock order")
        elif symmetry_score < 3.0:
            logger.info("Good symmetry: Model is mostly invariant to stock order")
        else:
            logger.info("Poor symmetry: Model performance depends on stock order")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    # 웹 서버 시작
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    logger.info("Web interface running at http://localhost:8000")
    
    # 메인 실행
    main()
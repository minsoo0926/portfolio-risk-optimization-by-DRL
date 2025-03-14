import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from generate_scenario import generate_scenario
import os
import time

# 사용자 정의 환경
class PortfolioEnv(gym.Env):
    def __init__(self, seed):
        super(PortfolioEnv, self).__init__()
        # 52-dim state (예: 10개 주식의 일단위 수익률, 63일 평균 수익률, 63일 표준편차, Relative Volume, VIX 지수, 5년 국채금리, previous action)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        # 10-dim action: 각 주식의 가중치 (-1~1 사이 값, 이후 정규화하여 합=0)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        # Load data for the current episode
        self.seed = seed
        data = None
        while data is None:
            data = generate_scenario(10, seed)
        
        # 첫 번째 열(날짜 등)을 제외한 데이터만 사용
        self.market_data = data.iloc[:, 1:41].values  # 주식 관련 데이터 (10종목 * 4features = 40)
        self.macro_data = data.iloc[:, 41:43].values  # VIX, 국채금리 등의 매크로 데이터 (2 features)
        
        self.max_steps = len(data)
        self.current_step = 0
        self.previous_action = np.zeros(10)  # 이전 행동은 0으로 초기화
        
        # 초기 상태 설정
        self.state = self._get_state()

    def _get_state(self):
        # 현재 시장 데이터와 이전 행동을 결합하여 상태 생성
        return np.concatenate([
            self.market_data[self.current_step].flatten(),  # 40 features
            self.macro_data[self.current_step].flatten(),   # 2 features
            self.previous_action                           # 10 features (이전 행동)
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.current_step = 0
        self.previous_action = np.zeros(10)
        self.state = self._get_state()
        return self.state, {}

    def step(self, action):
        # 행동 정규화: 가중치 합이 0이 되도록 함
        action = action - np.mean(action)
        weights = action / (np.sum(np.abs(action)) + 1e-8)
        
        # 수익률 데이터 추출 (각 주식의 일일 수익률은 첫 번째 feature라고 가정)
        returns_indices = np.arange(0, 40, 4)  # 0, 4, 8, ..., 36
        stock_returns = self.market_data[self.current_step, returns_indices]
        
        # 변동성 데이터 추출 (각 주식의 변동성은 세 번째 feature라고 가정)
        vol_indices = np.arange(2, 40, 4)      # 2, 6, 10, ..., 38
        stock_vols = self.market_data[self.current_step, vol_indices]
        
        # 포트폴리오 수익률 계산
        portfolio_return = np.sum(weights * stock_returns)
        
        # 포트폴리오 위험 계산 (간소화 버전: 상관관계 무시)
        portfolio_vol = np.sqrt(np.sum((weights * stock_vols) ** 2))
        
        # 턴오버 계산 (이전 가중치와의 차이)
        turnover = np.sum(np.abs(weights - self.previous_action))
        
        # 스케일링을 고려한 보상 계산 - 보상 함수 조정
        # 기존: reward = portfolio_return - 10 * portfolio_vol - 1 * turnover
        reward = 100 * portfolio_return - 5 * portfolio_vol - 0.5 * turnover
        
        # 이전 행동 업데이트
        self.previous_action = weights.copy()
        
        # 다음 단계로 이동
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # 일반적으로 시간 제한 초과 시 True
        
        # 새로운 상태 계산 (에피소드가 끝나지 않았을 경우)
        if not terminated:
            self.state = self._get_state()
        
        info = {
            "portfolio_return": portfolio_return,
            "portfolio_vol": portfolio_vol,
            "turnover": turnover
        }
        
        return self.state, reward, terminated, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self.current_step}, Portfolio weights: {self.previous_action}")
        return 1

# CustomCallback 클래스를 main 함수 밖으로 이동
class CustomCallback(BaseCallback):
    def __init__(self, eval_env, verbose=0, save_freq=10000, eval_freq=20000, model_path="ppo_portfolio"):
        super(CustomCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_results = []
        self.best_mean_reward = -np.inf
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.model_path = model_path
        
    def _on_step(self):
        try:
            # 주기적으로 모델 저장
            if self.num_timesteps % self.save_freq == 0:
                self.model.save(f"{self.model_path}_{self.num_timesteps}")
                print(f"Timestep {self.num_timesteps}: 모델 저장 완료 ({self.model_path}_{self.num_timesteps})")
            
            # 주기적으로 모델 성능 평가
            if self.num_timesteps % self.eval_freq == 0:
                self._evaluate_model()
                
                # 메모리 관리: 최근 10개의 결과만 유지
                if len(self.eval_results) > 10:
                    self.eval_results = self.eval_results[-10:]
                
            return True
        except Exception as e:
            print(f"콜백 에러 발생: {str(e)}")
            return False
            
    def _evaluate_model(self):
        print(f"\n===== Timestep {self.num_timesteps} 모델 평가 =====")
        episode_rewards = []
        episode_vols = []
        
        # 평가 환경에서 에피소드 실행
        obs, _ = self.eval_env.reset()  # 튜플에서 첫 번째 요소만 사용
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            episode_rewards.append(info["portfolio_return"])
            episode_vols.append(info["portfolio_vol"])
        
        # 성과 지표 계산
        mean_reward = np.mean(episode_rewards)
        mean_vol = np.mean(episode_vols)
        sharpe = mean_reward / (np.std(episode_rewards) + 1e-8)
        
        self.eval_results.append({
            "timestep": self.num_timesteps,
            "mean_reward": mean_reward,
            "mean_vol": mean_vol,
            "sharpe": sharpe
        })
        
        print(f"평균 수익률: {mean_reward:.4f}")
        print(f"평균 변동성: {mean_vol:.4f}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        
        # 최고 성능 모델 저장
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(f"{self.model_path}_best")
            print(f"새로운 최고 성능 모델 저장 ({self.model_path}_best), 수익률: {mean_reward:.4f}")

def main():
    try:
        model_path = "ppo_portfolio"
        total_timesteps = 500000  # 학습 스텝 증가
        eval_episodes = 100
        save_freq = 10000
        eval_freq = 20000

        # 환경 생성 시 시드 범위 제한
        train_env = PortfolioEnv(seed=0)
        
        # 저장된 모델이 있으면 불러오고, 없으면 새로 생성
        if os.path.exists(model_path + ".zip"):
            try:
                # 기존 모델 로드 시도
                model = PPO.load(model_path, env=train_env)
                print(f"모델 불러오기 완료 ({model_path})")
            except ValueError as e:
                # 관측 공간 불일치 오류 발생 시 새 모델 생성
                print(f"기존 모델 로드 실패: {e}")
                print("새 모델을 생성합니다.")
                model = PPO("MlpPolicy", train_env, 
                            learning_rate=1e-4,  # 학습률 감소
                            n_steps=2048, 
                            batch_size=128,  # 배치 크기 증가
                            gamma=0.99,
                            ent_coef=0.01,  # 엔트로피 계수 추가
                            clip_range=0.2,  # 클리핑 범위 명시
                            verbose=1)
        else:
            model = PPO("MlpPolicy", train_env, 
                        learning_rate=1e-4,  # 학습률 감소
                        n_steps=2048, 
                        batch_size=128,  # 배치 크기 증가
                        gamma=0.99,
                        ent_coef=0.01,  # 엔트로피 계수 추가
                        clip_range=0.2,  # 클리핑 범위 명시
                        verbose=1)
            print("새 모델 생성")
        
        # 평가용 환경 생성
        eval_env = PortfolioEnv(seed=9999)
        callback = CustomCallback(
            eval_env=eval_env,
            save_freq=save_freq,
            eval_freq=eval_freq,
            model_path=model_path
        )
        
        # 콜백과 함께 모델 학습
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # 최종 모델 저장
        model.save(model_path)
        print(f"최종 모델 저장 완료: {model_path}")
        
        # 학습된 모델 평가 (여러 시드로 테스트)
        results = []
        max_attempts = 3  # 최대 시도 횟수 설정
        
        for eval_seed in range(1000, 1000 + eval_episodes):
            attempts = 0
            while attempts < max_attempts:
                try:
                    # 평가용 환경 생성 (학습에 사용되지 않은 시드)
                    test_env = PortfolioEnv(seed=eval_seed)
                    
                    obs, _ = test_env.reset()  # 튜플에서 첫 번째 요소만 사용
                    done = False
                    episode_returns = []
                    episode_vols = []
                    
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = test_env.step(action)
                        done = terminated or truncated
                        episode_returns.append(info["portfolio_return"])
                        episode_vols.append(info["portfolio_vol"])
                    
                    # 성과 평가 지표 계산
                    mean_return = np.mean(episode_returns)
                    std_return = np.std(episode_returns) if len(episode_returns) > 1 else 1e-8
                    sharpe_ratio = mean_return / (std_return + 1e-8)
                    
                    results.append({
                        "seed": eval_seed,
                        "mean_return": mean_return,
                        "vol": np.mean(episode_vols),
                        "sharpe": sharpe_ratio
                    })
                    
                    break  # 성공적으로 완료되면 루프 종료
                    
                except Exception as e:
                    attempts += 1
                    print(f"시드 {eval_seed} 평가 실패 ({attempts}/{max_attempts}): {str(e)}")
                    if attempts >= max_attempts:
                        print(f"시드 {eval_seed} 평가 건너뜀")
            
            # 중간 결과 저장 (100개 에피소드마다)
            if len(results) % 100 == 0 and len(results) > 0:
                avg_sharpe = np.mean([r["sharpe"] for r in results])
                avg_return = np.mean([r["mean_return"] for r in results])
                avg_vol = np.mean([r["vol"] for r in results])
                
                print(f"\n===== 중간 평가 결과 ({len(results)} 에피소드) =====")
                print(f"평균 수익률: {avg_return:.4f}")
                print(f"평균 변동성: {avg_vol:.4f}")
                print(f"평균 Sharpe: {avg_sharpe:.4f}")
        
        # 최종 평가 결과 요약
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_return = np.mean([r["mean_return"] for r in results])
        avg_vol = np.mean([r["vol"] for r in results])
        
        print("\n===== 최종 평가 결과 =====")
        print(f"평균 수익률: {avg_return:.4f}")
        print(f"평균 변동성: {avg_vol:.4f}")
        print(f"평균 Sharpe: {avg_sharpe:.4f}")

    except Exception as e:
        print(f"학습 중 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()  # 상세 에러 정보 출력
        raise e

if __name__ == '__main__':
    iteration = 0
    
    while True:  # 무한 반복
        try:
            print(f"\n===== 학습 반복 #{iteration+1} 시작 =====")
            main()
            iteration += 1
            print(f"학습 반복 #{iteration} 완료")
            
            # 선택적: 일정 시간 대기 (서버 부하 방지)
            time.sleep(10)  # 10초 대기
            
        except Exception as e:
            print(f"실행 실패 (반복 #{iteration+1}): {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 에러 발생 시 잠시 대기 후 재시도
            time.sleep(60)  # 1분 대기
            
            # 선택적: 심각한 에러 발생 시 로그 기록
            with open("error_log.txt", "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 반복 #{iteration+1} 에러: {str(e)}\n")
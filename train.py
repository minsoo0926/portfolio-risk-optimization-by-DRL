import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from generate_scenario import generate_scenario
import os
import time
import torch

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
        
        # 보상 버퍼 초기화 (환경 생성 시 한 번만)
        self.reward_buffer = []
        self.return_buffer = []
        self.vol_buffer = []
        
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
        # 행동 정규화
        action = action - np.mean(action)
        weights = action / (np.sum(np.abs(action)) + 1e-8)
        
        # 현재 가중치 저장
        self.previous_action = weights.copy()
        
        # 다음 시점으로 이동
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 새 시점에서의 수익률 데이터 (t+1 시점)
        if not terminated:
            returns_indices = np.arange(0, 40, 4)
            stock_returns = self.market_data[self.current_step, returns_indices]
            
            vol_indices = np.arange(2, 40, 4)
            stock_vols = self.market_data[self.current_step, vol_indices]
            
            # 포트폴리오 수익률 계산 (이전 가중치 * 현재 수익률)
            portfolio_return = np.sum(weights * stock_returns)
            
            # 포트폴리오 위험 계산
            portfolio_vol = np.sqrt(np.sum((weights * stock_vols) ** 2))
            
            # 턴오버 계산 (다음 행동에서 계산되므로 여기서는 0)
            turnover = 0  # 첫 스텝에서는 이전 가중치가 없으므로 0
            
            # 보상 계산
            raw_reward = 5 * portfolio_return - 0.1 * portfolio_vol - 0.01 * turnover
            reward = raw_reward
            
            # 새로운 상태 계산
            self.state = self._get_state()
        else:
            # 에피소드 종료 시 보상 없음
            portfolio_return = 0
            portfolio_vol = 0
            turnover = 0
            reward = 0
        
        # info 딕셔너리에 수익률과 변동성 정보 포함
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
        
        # temp 디렉토리가 없으면 생성
        if not os.path.exists('temp'):
            os.makedirs('temp')
            print("temp 폴더를 생성했습니다.")
        
    def _on_step(self):
        try:
            # 주기적으로 모델 저장 (이제 temp 폴더에 저장)
            if self.num_timesteps % self.save_freq == 0:
                # temp 폴더에 중간 모델 저장
                temp_model_path = os.path.join('temp', f"{self.model_path}_{self.num_timesteps}")
                self.model.save(temp_model_path)
                print(f"Timestep {self.num_timesteps}: 모델 저장 완료 (temp/{self.model_path}_{self.num_timesteps})")
            
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
        print("\n" + "="*50)
        print(f"===== Timestep {self.num_timesteps} 모델 평가 =====")
        print("="*50)
        
        # 결과 저장용 변수
        daily_returns = []  # 원본 일일 수익률
        net_returns = []    # 거래비용 차감 후 순 수익률
        episode_vols = []
        risk_free_rates = []
        
        # 초기 포트폴리오 가치
        initial_capital = 10000
        portfolio_value = initial_capital
        
        # 평가 환경에서 에피소드 실행
        obs, _ = self.eval_env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            
            # 일일 수익률 및 기타 지표 저장
            daily_return = info["portfolio_return"] / 100.0  # 퍼센트를 소수점으로 변환
            daily_returns.append(daily_return)
            
            turnover = info["turnover"]
            transaction_cost = 0.001 * turnover  # 0.1% 거래비용
            
            # 거래비용 반영한 순 수익률
            net_return = daily_return - transaction_cost
            net_returns.append(net_return)
            
            # 포트폴리오 가치 업데이트 (복리 적용)
            portfolio_value *= (1 + net_return)
            
            episode_vols.append(info["portfolio_vol"])
            
            # 국채금리 데이터 수집
            if hasattr(self.eval_env, 'macro_data') and self.eval_env.current_step-1 < len(self.eval_env.macro_data):
                t_bill_rate = self.eval_env.macro_data[self.eval_env.current_step-1, 1] / 100.0
                risk_free_rates.append(t_bill_rate)
        
        # 일별 평균 수익률 (산술평균)
        mean_daily_return = np.mean(daily_returns) * 100  # 퍼센트로 변환
        mean_net_return = np.mean(net_returns) * 100      # 퍼센트로 변환
        
        # 복리 기준 전체 수익률 계산 (순 수익률 기준)
        total_return = (portfolio_value / initial_capital - 1) * 100
        
        # 투자 기간에 따른 연간화 조정
        days = len(daily_returns)
        trading_days_per_year = 252
        
        # 연간 환산 수익률 계산
        annualized_return = ((1 + total_return/100) ** (trading_days_per_year/days) - 1) * 100
        
        # 평균 변동성
        mean_vol = np.mean(episode_vols)
        
        # 무위험 이자율 계산 - 단순화
        if risk_free_rates:
            # 일별 이자율의 산술평균에 거래일수를 곱해 연율화
            annual_risk_free_rate = np.mean(risk_free_rates) * 100 * trading_days_per_year
        else:
            annual_risk_free_rate = 2.0  # 연 2% 기본값
        
        # 샤프 비율 계산 - 순 수익률 기준으로 표준편차 계산 (중요!)
        net_returns_array = np.array(net_returns) * 100  # 퍼센트로 변환
        daily_std = np.std(net_returns_array)  # 순 수익률의 표준편차
        annualized_std = daily_std * np.sqrt(trading_days_per_year)
        
        # 샤프 비율: (연간 수익률 - 연간 무위험 이자율) / 연간 표준편차
        sharpe = (annualized_return - annual_risk_free_rate) / (annualized_std + 1e-8)
        
        self.eval_results.append({
            "timestep": self.num_timesteps,
            "mean_daily_return": mean_daily_return,
            "mean_net_return": mean_net_return,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "mean_vol": mean_vol,
            "sharpe": sharpe,
            "annual_risk_free_rate": annual_risk_free_rate
        })
        
        # 결과 출력
        print("\n성과 지표:")
        print(f"► 일일 평균 수익률 (거래비용 전): {mean_daily_return:.4f}%")
        print(f"► 일일 평균 순수익률 (거래비용 후): {mean_net_return:.4f}%")
        print(f"► 복리 계산 총 수익률: {total_return:.4f}%")
        print(f"► 연간 환산 수익률: {annualized_return:.4f}%")
        print(f"► 연간 무위험 이자율: {annual_risk_free_rate:.4f}%")
        print(f"► 일별 수익률 표준편차(연간화): {annualized_std:.4f}%")
        print(f"► Sharpe Ratio: {sharpe:.4f}")
        
        # 로그 기록 부분도 수정
        with open("evaluation_log.txt", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Timestep {self.num_timesteps}\n")
            f.write(f"일일 평균 수익률 (거래비용 전): {mean_daily_return:.4f}%\n")
            f.write(f"일일 평균 순수익률 (거래비용 후): {mean_net_return:.4f}%\n")
            f.write(f"복리 계산 총 수익률: {total_return:.4f}%\n")
            f.write(f"연간 환산 수익률: {annualized_return:.4f}%\n")
            f.write(f"연간 무위험 이자율: {annual_risk_free_rate:.4f}%\n")
            f.write(f"일별 수익률 표준편차(연간화): {annualized_std:.4f}%\n")
            f.write(f"Sharpe Ratio: {sharpe:.4f}\n\n")
        
        # 최고 성능 모델 저장 (이제는 복리 수익률 기준으로)
        if total_return > self.best_mean_reward:
            self.best_mean_reward = total_return
            self.model.save(f"{self.model_path}_best")
            print(f"새로운 최고 성능 모델 저장 ({self.model_path}_best), 총 수익률: {total_return:.4f}%")

def main():
    try:
        model_path = "ppo_portfolio"
        total_timesteps = 500000  # 학습 스텝 증가
        eval_episodes = 100
        save_freq = 10000
        eval_freq = 20000

        # 학습 데이터의 다양성 확보
        train_seeds = np.random.randint(0, 10000, size=10)  # 여러 시드 사용
        
        # 여러 환경에서 학습
        train_envs = [PortfolioEnv(seed=seed) for seed in train_seeds]
        
        # 저장된 모델이 있으면 불러오고, 없으면 새로 생성
        if os.path.exists(model_path + ".zip"):
            try:
                # 기존 모델 로드 시도
                model = PPO.load(model_path, env=train_envs[0])
                print(f"모델 불러오기 완료 ({model_path})")
            except ValueError as e:
                # 관측 공간 불일치 오류 발생 시 새 모델 생성
                print(f"기존 모델 로드 실패: {e}")
                print("새 모델을 생성합니다.")
                policy_kwargs = dict(
                    net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])],
                    activation_fn=torch.nn.ReLU
                )
                model = PPO("MlpPolicy", train_envs[0], policy_kwargs=policy_kwargs,
                            learning_rate=0.0001,      # 학습률 설정
                            n_steps=2048,              # 스텝 수 설정
                            batch_size=128,            # 배치 크기 설정
                            gamma=0.99,                # 감마 설정
                            ent_coef=0.02,             # 엔트로피 계수 설정
                            clip_range=0.1,            # 클리핑 범위 설정
                            vf_coef=0.7,               # 가치 함수 계수 설정
                            max_grad_norm=0.3,         # 최대 그래디언트 노름 설정
                            verbose=2)
        else:
            policy_kwargs = dict(
                net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])],
                activation_fn=torch.nn.ReLU
            )
            model = PPO("MlpPolicy", train_envs[0], policy_kwargs=policy_kwargs,
                        learning_rate=0.0001,
                        n_steps=2048,
                        batch_size=128,
                        gamma=0.99,
                        ent_coef=0.02,
                        clip_range=0.1,
                        vf_coef=0.7,
                        max_grad_norm=0.3,
                        verbose=2)
            print("새 모델 생성")
        
        # 평가용 환경 생성
        eval_env = PortfolioEnv(seed=9999)
        callback = CustomCallback(
            eval_env=eval_env,
            save_freq=5000,   # 더 자주 저장
            eval_freq=5000,   # 더 자주 평가
            model_path=model_path
        )
        
        # 각 환경에서 번갈아가며 학습
        learning_steps_per_env = 5000
        for cycle in range(total_timesteps // (len(train_envs) * learning_steps_per_env)):
            print(f"\n===== 학습 사이클 {cycle+1} 시작 =====")
            
            for env_idx, env in enumerate(train_envs):
                try:
                    model.set_env(env)  # 환경 변경
                    
                    # 마지막 환경에서만 콜백 사용
                    if env_idx == len(train_envs) - 1:
                        model.learn(total_timesteps=learning_steps_per_env, 
                                   reset_num_timesteps=False, 
                                   callback=callback)
                    else:
                        model.learn(total_timesteps=learning_steps_per_env,
                                   reset_num_timesteps=False)
                    
                    print(f"환경 {env_idx} (시드 {train_seeds[env_idx]}) 학습 완료")
                    
                    # 각 환경 학습 후 별도 평가 실행
                    if env_idx % 3 == 0:  # 3개 환경마다 한 번씩 평가
                        print("\n----- 환경 학습 후 중간 평가 -----")
                        obs, _ = eval_env.reset()
                        done = False
                        rewards = []
                        while not done:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, terminated, truncated, info = eval_env.step(action)
                            done = terminated or truncated
                            rewards.append(info["portfolio_return"])
                        print(f"평균 수익률: {np.mean(rewards):.4f}")
                    
                except Exception as e:
                    print(f"환경 {env_idx} 학습 중 오류: {e}")
                    continue
            
            # 각 사이클 후 강제 평가
            if cycle > 0 and cycle % 2 == 0:  # 2 사이클마다 한 번씩
                callback._evaluate_model()
        
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
                    
                    obs, _ = test_env.reset()
                    done = False
                    daily_returns = []
                    net_returns = []
                    episode_vols = []
                    t_bill_rates = []
                    
                    # 초기 자본
                    initial_capital = 10000
                    portfolio_value = initial_capital
                    
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = test_env.step(action)
                        done = terminated or truncated
                        
                        # 일일 수익률 및 기타 지표 저장
                        daily_return = info["portfolio_return"] / 100.0
                        daily_returns.append(daily_return)
                        
                        turnover = info["turnover"]
                        transaction_cost = 0.001 * turnover
                        
                        # 거래비용 반영한 순 수익률
                        net_return = daily_return - transaction_cost
                        net_returns.append(net_return)
                        
                        # 포트폴리오 가치 업데이트 (복리 적용)
                        portfolio_value *= (1 + net_return)
                        
                        episode_vols.append(info["portfolio_vol"])
                        
                        # 국채금리 데이터 수집
                        if test_env.current_step-1 < len(test_env.macro_data):
                            t_bill_rate = test_env.macro_data[test_env.current_step-1, 1] / 100.0
                            t_bill_rates.append(t_bill_rate)
                    
                    # 복리 기준 전체 수익률 계산
                    total_return = (portfolio_value / initial_capital - 1) * 100
                    
                    # 연간 환산 수익률 계산
                    days = len(daily_returns)
                    trading_days_per_year = 252
                    annualized_return = ((1 + total_return/100) ** (trading_days_per_year/days) - 1) * 100
                    
                    # 무위험 이자율 계산
                    if t_bill_rates:
                        annual_risk_free_rate = np.mean(t_bill_rates) * 100 * trading_days_per_year
                    else:
                        annual_risk_free_rate = 2.0
                    
                    # 샤프 비율 계산 - 순 수익률 기준 표준편차
                    net_returns_array = np.array(net_returns) * 100
                    daily_std = np.std(net_returns_array)
                    annualized_std = daily_std * np.sqrt(trading_days_per_year)
                    sharpe_ratio = (annualized_return - annual_risk_free_rate) / (annualized_std + 1e-8)
                    
                    results.append({
                        "seed": eval_seed,
                        "daily_return": daily_return,
                        "total_return": total_return,
                        "annual_return": annualized_return,
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
                avg_return = np.mean([r["total_return"] for r in results])
                avg_vol = np.mean([r["vol"] for r in results])
                
                print(f"\n===== 중간 평가 결과 ({len(results)} 에피소드) =====")
                print(f"평균 수익률: {avg_return:.4f}")
                print(f"평균 변동성: {avg_vol:.4f}")
                print(f"평균 Sharpe: {avg_sharpe:.4f}")
        
        # 최종 평가 결과 요약
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_return = np.mean([r["total_return"] for r in results])
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
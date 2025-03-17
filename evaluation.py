import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO
from train import PortfolioEnv
import os
import seaborn as sns
from datetime import datetime

# 시각화 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def evaluate_model(model_path, seed=1234, initial_capital=10000, debug=True):
    """
    학습된 모델을 평가하고 포트폴리오 가치 추이를 시각화합니다.
    
    Args:
        model_path: 학습된 모델 경로
        seed: 평가에 사용할 시드
        initial_capital: 초기 투자 금액
        debug: 디버깅 정보 출력 여부
    """
    # 환경 생성
    env = PortfolioEnv(seed=seed)
    
    # 모델 로드
    model = PPO.load(model_path)
    print(f"모델 로드 완료: {model_path}")
    
    # 평가 실행
    obs, _ = env.reset()
    done = False
    
    # 결과 저장용 리스트
    dates = []
    portfolio_values = []
    returns = []
    vols = []
    weights_history = []
    turnovers = []  # 여기서 turnovers 리스트 초기화
    
    # 날짜 데이터 생성 (시나리오에서 사용된 날짜 정보가 없으므로 임의로 생성)
    start_date = datetime(2022, 1, 1)
    
    # 에피소드 실행
    step = 0
    portfolio_value = initial_capital
    
    while not done:
        # 모델의 행동 예측
        action, _ = model.predict(obs, deterministic=True)
        
        # 행동 정규화: 가중치 합이 0이 되도록 함
        action = action - np.mean(action)
        action = action / (np.sum(np.abs(action)) + 1e-8)
        
        # 환경에서 한 스텝 진행
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 날짜 추가
        current_date = start_date + pd.Timedelta(days=step)
        dates.append(current_date)
        
        # 포트폴리오 가치 계산 및 추가
        daily_return = info["portfolio_return"] / 100.0
        turnover = info["turnover"]
        turnovers.append(turnover)  # 턴오버를 리스트에 추가
        
        # 거래비용 계산 (턴오버에 비례하는 거래비용)
        transaction_cost = 0.001 * turnover
        
        # 순 수익률 계산 및 포트폴리오 가치 업데이트
        net_return = daily_return - transaction_cost
        portfolio_value *= (1 + net_return)
        
        portfolio_values.append(portfolio_value)
        
        # 수익률과 변동성 저장
        returns.append(daily_return)
        vols.append(info["portfolio_vol"])
        
        # 가중치 저장
        weights_history.append(env.previous_action)
        
        # 첫 번째 스텝에서 자세한 디버깅 정보 출력
        if step == 0:
            print("\n===== 첫 번째 스텝 디버깅 =====")
            print(f"원본 일별 수익률: {daily_return * 100:.4f}%")
            print(f"스케일 조정 후 수익률: {daily_return:.6f}")
            print(f"거래비용: {transaction_cost:.6f}")
            print(f"순 수익률: {net_return:.6f}")
            print(f"포트폴리오 가치 변화: {initial_capital:.2f} -> {portfolio_value:.2f}")
        
        step += 1
    
    # 디버깅 정보 출력
    if debug:
        print("\n===== 디버깅 정보 =====")
        print(f"첫 10개 일별 수익률: {returns[:10]}")
        print(f"첫 10개 포트폴리오 가치: {portfolio_values[:10]}")
        print(f"첫 10개 변동성: {vols[:10]}")
        print(f"첫 10개 가중치: {weights_history[:10]}")
        
        # 수익률 통계
        print(f"\n수익률 통계:")
        print(f"최소: {min(returns):.4f}%, 최대: {max(returns):.4f}%, 평균: {np.mean(returns):.4f}%")
        print(f"표준편차: {np.std(returns):.4f}%")
        
        # 이상치 확인
        outliers = [r for r in returns if abs(r) > 5.0]  # 5% 이상의 일별 수익률은 이상치로 간주
        if outliers:
            print(f"\n이상치 수익률 ({len(outliers)}개): {outliers}")
        
        # 데이터 스케일 확인
        print("\n===== 데이터 스케일 확인 =====")
        stock_returns_scale = []
        for i in range(10):  # 10개 주식 각각의 수익률 범위 확인
            returns_idx = i * 4  # 수익률은 첫 번째 특성
            stock_returns = [env.market_data[j, returns_idx] for j in range(len(env.market_data))]
            print(f"주식 {i+1} 수익률 범위: {min(stock_returns):.4f} ~ {max(stock_returns):.4f}, 평균: {np.mean(stock_returns):.4f}")
            stock_returns_scale.append((min(stock_returns), max(stock_returns), np.mean(stock_returns)))
        
        # 원본 시장 데이터의 일부 샘플 출력
        print("\n원본 시장 데이터 샘플 (처음 3일):")
        for day in range(min(3, len(env.market_data))):
            print(f"Day {day+1}: {env.market_data[day, :10]}")  # 처음 10개 값만 출력
    
    # 결과 데이터프레임 생성
    results = pd.DataFrame({
        'Date': dates,
        'Portfolio_Value': portfolio_values,
        'Daily_Return': returns,
        'Volatility': vols
    })
    
    # 누적 수익률 계산
    results['Cumulative_Return'] = (results['Portfolio_Value'] / initial_capital - 1) * 100
    
    # 일별 수익률 로그를 CSV 파일로 저장
    daily_returns_df = pd.DataFrame({
        'Date': dates,
        'Daily_Return': returns,
        'Net_Return': [r - 0.001 * t/100.0 for r, t in zip(returns, turnovers)],
        'Portfolio_Value': portfolio_values
    })
    daily_returns_df.to_csv('daily_returns_log.csv', index=False)
    print(f"\n일별 수익률 로그가 'daily_returns_log.csv' 파일로 저장되었습니다.")
    
    # 성과 지표 계산
    total_return = (portfolio_values[-1] / initial_capital - 1) * 100
    annualized_return = ((1 + total_return/100) ** (252/len(portfolio_values)) - 1) * 100
    annualized_vol = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # 결과 출력
    print(f"\n===== 모델 평가 결과 =====")
    print(f"시드: {seed}")
    print(f"총 수익률: {total_return:.2f}%")
    print(f"연율화 수익률: {annualized_return:.2f}%")
    print(f"연율화 변동성: {annualized_vol:.2f}%")
    print(f"샤프 비율: {sharpe_ratio:.2f}")
    print(f"최대 낙폭: {max_drawdown:.2f}%")
    
    # 포트폴리오 가치 추이 시각화
    plt.figure(figsize=(14, 10))
    
    # 포트폴리오 가치 차트
    plt.subplot(2, 1, 1)
    plt.plot(results['Date'], results['Portfolio_Value'], 'b-', linewidth=2)
    plt.title('Portfolio Value Over Time', fontsize=15)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # 일별 수익률 차트
    plt.subplot(2, 1, 2)
    plt.plot(results['Date'], results['Daily_Return'], 'g-', linewidth=1)
    plt.title('Daily Returns', fontsize=15)
    plt.ylabel('Return (%)', fontsize=12)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    plt.show()
    
    # 주식별 가중치 추이 시각화
    plt.figure(figsize=(14, 8))
    weights_array = np.array(weights_history)
    
    for i in range(weights_array.shape[1]):
        plt.plot(dates, weights_array[:, i], label=f'Stock {i+1}')
    
    plt.title('Portfolio Weights Over Time', fontsize=15)
    plt.ylabel('Weight', fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('portfolio_weights.png')
    plt.show()
    
    return results, weights_array

def calculate_max_drawdown(portfolio_values):
    """최대 낙폭(Maximum Drawdown) 계산"""
    peak = portfolio_values[0]
    max_dd = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return max_dd

def compare_models(model_paths, seeds=[1234, 5678, 9012], initial_capital=10000):
    """여러 모델의 성능을 비교합니다."""
    all_results = []
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        
        # 여러 시드에 대해 평가
        seed_results = []
        for seed in seeds:
            env = PortfolioEnv(seed=seed)
            model = PPO.load(model_path)
            
            obs, _ = env.reset()
            done = False
            
            portfolio_value = initial_capital
            portfolio_values = [portfolio_value]
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                
                # 행동 정규화: 환경의 step 메서드와 동일하게 처리
                action = action - np.mean(action)
                action = action / (np.sum(np.abs(action)) + 1e-8)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 일별 수익률 계산 - 100으로 나누어 소수점 단위로 변환
                daily_return = info["portfolio_return"] / 100.0
                turnover = info["turnover"]
                
                # 거래비용 계산
                transaction_cost = 0.001 * turnover
                
                # 순 수익률 적용
                net_return = daily_return - transaction_cost
                portfolio_value *= (1 + net_return)
                
                portfolio_values.append(portfolio_value)
            
            # 성과 지표 계산
            total_return = (portfolio_values[-1] / initial_capital - 1) * 100
            seed_results.append(total_return)
        
        # 평균 및 표준편차 계산
        avg_return = np.mean(seed_results)
        std_return = np.std(seed_results)
        
        all_results.append({
            'Model': model_name,
            'Avg_Return': avg_return,
            'Std_Return': std_return,
            'Min_Return': min(seed_results),
            'Max_Return': max(seed_results)
        })
    
    # 결과 데이터프레임 생성 및 출력
    results_df = pd.DataFrame(all_results)
    print("\n===== 모델 비교 결과 =====")
    print(results_df)
    
    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['Model'], results_df['Avg_Return'], yerr=results_df['Std_Return'], 
            capsize=5, color='skyblue', alpha=0.7)
    plt.title('Average Return by Model', fontsize=15)
    plt.ylabel('Average Return (%)', fontsize=12)
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    return results_df

def robust_evaluation(model_path, seeds=range(1000, 1100), initial_capital=10000):
    """다양한 시드에서 모델의 강건성을 평가합니다."""
    results = []
    
    for seed in seeds:
        try:
            env = PortfolioEnv(seed=seed)
            model = PPO.load(model_path)
            
            obs, _ = env.reset()
            done = False
            
            portfolio_value = initial_capital
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = action - np.mean(action)
                action = action / (np.sum(np.abs(action)) + 1e-8)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                daily_return = info["portfolio_return"] / 100.0
                turnover = info["turnover"]
                transaction_cost = 0.001 * turnover
                
                net_return = daily_return - transaction_cost
                portfolio_value *= (1 + net_return)
            
            total_return = (portfolio_value / initial_capital - 1) * 100
            results.append(total_return)
            print(f"시드 {seed}: 수익률 {total_return:.2f}%")
        
        except Exception as e:
            print(f"시드 {seed} 평가 중 오류: {e}")
            continue
    
    # 결과 요약
    if results:
        print(f"\n===== 강건성 평가 결과 ({len(results)}개 시드) =====")
        print(f"평균 수익률: {np.mean(results):.2f}%")
        print(f"수익률 표준편차: {np.std(results):.2f}%")
        print(f"최소 수익률: {min(results):.2f}%")
        print(f"최대 수익률: {max(results):.2f}%")
        
        # 히스토그램으로 시각화
        plt.figure(figsize=(10, 6))
        plt.hist(results, bins=20, alpha=0.7)
        plt.title('Return Distribution Across Different Seeds')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('return_distribution.png')
        plt.show()
    else:
        print("평가 결과가 없습니다.")
    
    return results

if __name__ == "__main__":
    # 단일 모델 평가
    model_path = "ppo_portfolio_best"  # 또는 "ppo_portfolio_best"
    results, weights = evaluate_model(model_path, seed=1234)
    
    # 여러 모델 비교 (선택적)
    # model_paths = ["ppo_portfolio", "ppo_portfolio_best", "ppo_portfolio_100000"]
    # compare_results = compare_models(model_paths)

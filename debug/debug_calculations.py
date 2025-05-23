import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import PortfolioEnv
from stable_baselines3 import PPO
import logging
import sys
from app.utils import setup_logger

logger = setup_logger()

def debug_sharpe_calculation(returns, trading_days_per_year=252, risk_free_rate=2.0, transaction_costs=0.001, turnovers=None):
    """
    Sharpe 비율 계산 과정을 상세히 디버깅하는 함수
    
    Args:
        returns: 일별 수익률 리스트 (소수점 형태, 예: 0.01 = 1%)
        trading_days_per_year: 연간 거래일 수
        risk_free_rate: 연간 무위험 이자율 (%)
        transaction_costs: 거래 비용 비율 (0.001 = 0.1%)
        turnovers: 일별 턴오버 리스트 (없으면 모든 날에 50% 턴오버 가정)
    """
    logger.info("="*50)
    logger.info("Sharpe 비율 계산 디버깅")
    logger.info("="*50)
    
    # 입력 데이터 정보
    logger.info(f"입력 데이터: {len(returns)}일 수익률")
    logger.info(f"첫 10개 일별 수익률 (소수점): {returns[:10]}")
    
    # 퍼센트로 변환
    returns_pct = [r * 100 for r in returns]
    logger.info(f"첫 10개 일별 수익률 (퍼센트): {returns_pct[:10]}")
    
    # 턴오버가 없으면 기본값 설정
    if turnovers is None:
        turnovers = [50.0] * len(returns)  # 50% 턴오버 가정
    
    logger.info(f"첫 10개 일별 턴오버: {turnovers[:10]}")
    
    # 거래비용 계산
    transaction_costs_daily = [transaction_costs * t for t in turnovers]
    logger.info(f"첫 10개 일별 거래비용 (소수점): {transaction_costs_daily[:10]}")
    
    # 순 수익률 계산 (1): 직접 계산
    net_returns_method1 = [r - transaction_costs * t for r, t in zip(returns, turnovers)]
    logger.info(f"첫 10개 순 수익률 (방법 1, 소수점): {net_returns_method1[:10]}")
    
    # 순 수익률 계산 (2): 거래비용을 퍼센트로 변환하여 계산
    net_returns_method2 = [r - transaction_costs * t/100.0 for r, t in zip(returns, turnovers)]
    logger.info(f"첫 10개 순 수익률 (방법 2, 소수점): {net_returns_method2[:10]}")
    
    # 포트폴리오 가치 변화 계산
    initial_capital = 10000
    
    # 방법 1: 복리로 계산
    portfolio_value_method1 = initial_capital
    portfolio_values1 = [initial_capital]
    
    for net_return in net_returns_method1:
        portfolio_value_method1 *= (1 + net_return)
        portfolio_values1.append(portfolio_value_method1)
    
    total_return_method1 = (portfolio_value_method1 / initial_capital - 1) * 100
    logger.info(f"최종 포트폴리오 가치 (방법 1): {portfolio_value_method1:.2f}")
    logger.info(f"총 수익률 (방법 1, %): {total_return_method1:.4f}%")
    
    # 방법 2: 복리로 계산
    portfolio_value_method2 = initial_capital
    portfolio_values2 = [initial_capital]
    
    for net_return in net_returns_method2:
        portfolio_value_method2 *= (1 + net_return)
        portfolio_values2.append(portfolio_value_method2)
    
    total_return_method2 = (portfolio_value_method2 / initial_capital - 1) * 100
    logger.info(f"최종 포트폴리오 가치 (방법 2): {portfolio_value_method2:.2f}")
    logger.info(f"총 수익률 (방법 2, %): {total_return_method2:.4f}%")
    
    # 연간화된 수익률 계산
    days = len(returns)
    annualized_return_method1 = ((1 + total_return_method1/100) ** (trading_days_per_year/days) - 1) * 100
    annualized_return_method2 = ((1 + total_return_method2/100) ** (trading_days_per_year/days) - 1) * 100
    
    logger.info(f"연간화된 수익률 (방법 1, %): {annualized_return_method1:.4f}%")
    logger.info(f"연간화된 수익률 (방법 2, %): {annualized_return_method2:.4f}%")
    
    # 표준편차 계산
    # 방법 1: 일별 수익률의 표준편차 (소수점)
    daily_std1 = np.std(net_returns_method1)
    annualized_std1 = daily_std1 * np.sqrt(trading_days_per_year)
    
    # 방법 2: 일별 수익률의 표준편차 (퍼센트로 변환)
    net_returns_array = np.array(net_returns_method2) * 100
    daily_std2 = np.std(net_returns_array)
    annualized_std2 = daily_std2 * np.sqrt(trading_days_per_year)
    
    logger.info(f"일별 순 수익률 표준편차 (방법 1, 소수점): {daily_std1:.6f}")
    logger.info(f"일별 순 수익률 표준편차 (방법 2, 퍼센트): {daily_std2:.4f}%")
    
    logger.info(f"연간화된 표준편차 (방법 1): {annualized_std1:.6f}")
    logger.info(f"연간화된 표준편차 (방법 2): {annualized_std2:.4f}%")
    
    # 샤프 비율 계산
    sharpe1 = (annualized_return_method1 - risk_free_rate) / (annualized_std1 * 100)
    sharpe2 = (annualized_return_method2 - risk_free_rate) / annualized_std2
    
    logger.info("\n최종 Sharpe 비율 계산:")
    logger.info(f"방법 1: ({annualized_return_method1:.4f} - {risk_free_rate:.4f}) / {annualized_std1 * 100:.4f} = {sharpe1:.4f}")
    logger.info(f"방법 2: ({annualized_return_method2:.4f} - {risk_free_rate:.4f}) / {annualized_std2:.4f} = {sharpe2:.4f}")
    
    # 기존 코드의 계산 방법 (train.py에서의 방법)
    sharpe_train = (annualized_return_method2 - risk_free_rate) / (annualized_std2 + 1e-8)
    logger.info(f"train.py 방법: ({annualized_return_method2:.4f} - {risk_free_rate:.4f}) / ({annualized_std2:.4f} + 1e-8) = {sharpe_train:.4f}")
    
    # 시각화
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values1, label='방법 1: 일반 턴오버', color='blue')
    plt.plot(portfolio_values2, label='방법 2: 퍼센트 기준 턴오버', color='red')
    plt.title('포트폴리오 가치 변화 비교')
    plt.xlabel('날짜')
    plt.ylabel('포트폴리오 가치 ($)')
    plt.legend()
    plt.grid(True)
    
    # 일별 순 수익률 비교
    plt.subplot(2, 1, 2)
    plt.plot(net_returns_method1, label='방법 1: 일반 턴오버', color='blue', alpha=0.7)
    plt.plot(net_returns_method2, label='방법 2: 퍼센트 기준 턴오버', color='red', alpha=0.7)
    plt.title('일별 순 수익률 비교')
    plt.xlabel('날짜')
    plt.ylabel('순 수익률')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('debug_comparison.png')
    logger.info("비교 차트가 'debug_comparison.png' 파일로 저장되었습니다.")
    
    return {
        'total_return_method1': total_return_method1,
        'total_return_method2': total_return_method2,
        'annualized_return_method1': annualized_return_method1,
        'annualized_return_method2': annualized_return_method2,
        'annualized_std1': annualized_std1,
        'annualized_std2': annualized_std2,
        'sharpe1': sharpe1,
        'sharpe2': sharpe2,
        'sharpe_train': sharpe_train
    }

def evaluate_model_debug(model_path, seed=1234):
    """
    학습된 모델의 평가 결과를 디버깅하는 함수
    """
    logger.info("="*50)
    logger.info(f"모델 평가 디버깅 시작: {model_path} (시드: {seed})")
    logger.info("="*50)
    
    # 환경 생성
    env = PortfolioEnv(seed=seed)
    
    # 모델 로드
    model = PPO.load(model_path)
    logger.info(f"모델 로드 완료: {model_path}")
    
    # 평가 실행
    obs, _ = env.reset()
    done = False
    
    # 결과 저장용 리스트
    returns = []  # 원본 일일 수익률
    turnovers = []  # 일별 턴오버
    
    # 초기 포트폴리오 가치
    initial_capital = 10000
    
    # 에피소드 실행
    logger.info("시뮬레이션 시작...")
    while not done:
        try:
            # 모델의 행동 예측
            action, _ = model.predict(obs, deterministic=True)
            
            # 환경에서 한 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            # 폴백 액션 제공 (제로 웨이트)
            action = np.zeros(env.action_space.shape)
            
            # 폴백 액션으로 스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # 일별 수익률 및 턴오버 저장
        daily_return = info["portfolio_return"] / 100.0  # 퍼센트를 소수점으로 변환
        returns.append(daily_return)
        
        turnover = info["turnover"]
        turnovers.append(turnover)
    
    logger.info(f"시뮬레이션 완료: 총 {len(returns)}일 실행")
    
    # Sharpe 비율 계산 디버깅
    debug_results = debug_sharpe_calculation(returns, turnovers=turnovers)
    
    return debug_results

if __name__ == "__main__":
    model_path = "ppo_portfolio_best"  # 평가할 모델 경로
    evaluate_model_debug(model_path, seed=1234)
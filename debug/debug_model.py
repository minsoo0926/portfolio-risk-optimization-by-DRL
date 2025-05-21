from train import PortfolioEnv, NormalizedActorCriticPolicy
from stable_baselines3 import PPO
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import time

print("Running model testing...")

# Check if a trained model exists
model_path = "ppo_portfolio_best"
if os.path.exists(model_path + ".zip"):
    print(f"Found existing model at {model_path}.zip")
    # Test environment
    test_env = PortfolioEnv(seed=1234)
    model = PPO.load(model_path, custom_objects={"policy_class": NormalizedActorCriticPolicy})
    
    # Test the model
    obs, _ = test_env.reset()
    done = False
    portfolio_values = [10000]  # Initial capital
    returns = []
    net_returns = []
    weights_history = []
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        weights = action
        
        # Store weights
        weights_history.append(weights)
        
        # Take a step
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        # Get return information
        daily_return = info["portfolio_return"]  # Already in decimal form (e.g., 0.01 = 1%)
        turnover = info["turnover"]
        transaction_cost = 0.001 * turnover
        
        # Calculate net return after transaction costs
        net_return = daily_return - transaction_cost
        
        # Update portfolio value (compound returns)
        portfolio_values.append(portfolio_values[-1] * (1 + net_return))
        
        # Store returns
        returns.append(daily_return)
        net_returns.append(net_return)
        
        steps += 1
        
        # Print detailed info for first 5 steps
        if steps <= 5:
            print(f"\nStep {steps}:")
            print(f"  Raw daily return: {daily_return:.6f} ({daily_return*100:.4f}%)")
            print(f"  Portfolio weights: {weights.round(4)}")
            print(f"  Turnover: {turnover:.4f}")
            print(f"  Transaction cost: {transaction_cost:.6f}")
            print(f"  Net return: {net_return:.6f} ({net_return*100:.4f}%)")
            print(f"  Portfolio value: {portfolio_values[-1]:.2f}")
    
    # Calculate performance metrics
    total_days = len(returns)
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    
    # Trading days per year (standard assumption)
    trading_days_per_year = 252
    
    # Annualized return using compound formula
    annualized_return = ((1 + total_return/100) ** (trading_days_per_year/total_days) - 1) * 100
    
    # Calculate standard deviation of returns
    daily_std = np.std(net_returns) * 100  # Convert to percentage
    annualized_std = daily_std * np.sqrt(trading_days_per_year)
    
    # Annual risk-free rate (assumed 2% as in train.py)
    annual_risk_free_rate = 2.0
    
    # Sharpe ratio
    sharpe_ratio = (annualized_return - annual_risk_free_rate) / annualized_std
    
    # Maximum drawdown
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100
        if dd > max_drawdown:
            max_drawdown = dd
    
    # Print performance summary
    print("\n===== Model Performance Summary =====")
    print(f"Total days: {total_days}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    print(f"Daily standard deviation: {daily_std:.4f}%")
    print(f"Annualized standard deviation: {annualized_std:.2f}%")
    print(f"Sharpe ratio: {sharpe_ratio:.4f}")
    print(f"Maximum drawdown: {max_drawdown:.2f}%")
    
    # Return distribution
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(net_returns)*100, bins=30, alpha=0.7)
    plt.title('Distribution of Daily Net Returns')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('debug_return_distribution.png')
    print("\nReturn distribution saved as 'debug_return_distribution.png'")
    
    # Portfolio value over time
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig('debug_portfolio_value.png')
    print("Portfolio value chart saved as 'debug_portfolio_value.png'")
    
    # Calculate daily returns from portfolio values for verification
    pct_changes = [(portfolio_values[i] / portfolio_values[i-1] - 1) for i in range(1, len(portfolio_values))]
    
    # Verify that our net_returns match the percent changes in portfolio values
    print("\nVerifying calculations:")
    print(f"Mean of net_returns: {np.mean(net_returns)*100:.6f}%")
    print(f"Mean of portfolio value percent changes: {np.mean(pct_changes)*100:.6f}%")
    print(f"Difference: {(np.mean(net_returns) - np.mean(pct_changes))*100:.8f}%")  # Should be very close to zero
    
else:
    print(f"No model found at {model_path}.zip")
    print("Please run train.py first to create a model")
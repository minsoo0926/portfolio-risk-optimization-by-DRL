import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO
from train import PortfolioEnv
import os
import seaborn as sns
from datetime import datetime
import logging
import traceback
import threading
import time
from app.server import start_server

# Get logger
logger = logging.getLogger("portfolio_optimization")

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def evaluate_model(model_path, seed=1234, initial_capital=10000, debug=True):
    """
    Evaluates the trained model and visualizes portfolio value trends.
    
    Args:
        model_path: Path to the trained model
        seed: Seed to use for evaluation
        initial_capital: Initial investment amount
        debug: Whether to output debugging information
    """
    # Create environment
    logger.info(f"Starting model evaluation: {model_path} (seed: {seed})")
    env = PortfolioEnv(seed=seed)
    
    # Load model
    model = PPO.load(model_path)
    logger.info(f"Model loaded successfully: {model_path}")
    
    # Run evaluation
    obs, _ = env.reset()
    done = False
    
    # Lists for storing results
    dates = []
    portfolio_values = []
    returns = []
    vols = []
    weights_history = []
    turnovers = []  # Initialize turnovers list here
    
    # Create date data (arbitrary creation since scenario doesn't have date information)
    start_date = datetime(2022, 1, 1)
    
    # Run episode
    step = 0
    portfolio_value = initial_capital
    
    logger.info("Starting simulation...")
    while not done:
        # Predict model action
        action, _ = model.predict(obs, deterministic=True)
        
        # Normalize action: make weights sum to 0
        action = action - np.mean(action)
        action = action / (np.sum(np.abs(action)) + 1e-8)
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Add date
        current_date = start_date + pd.Timedelta(days=step)
        dates.append(current_date)
        
        # Calculate and add portfolio value
        daily_return = info["portfolio_return"]  # Already in decimal form (e.g., 0.01 = 1%)
        turnover = info["turnover"]
        turnovers.append(turnover)  # Add turnover to the list
        
        # Calculate transaction cost (proportional to turnover)
        transaction_cost = 0.001 * turnover
        
        # Calculate net return and update portfolio value
        net_return = daily_return - transaction_cost
        portfolio_value *= (1 + net_return)
        
        portfolio_values.append(portfolio_value)
        
        # Store returns and volatility
        returns.append(daily_return)
        vols.append(info["portfolio_vol"])
        
        # Store weights
        weights_history.append(env.previous_action)
        
        # Output detailed debugging info for the first step
        if step == 0:
            logger.info("\n===== First Step Debugging =====")
            logger.info(f"Original daily return: {daily_return * 100:.4f}%")
            logger.info(f"Return after scale adjustment: {daily_return:.6f}")
            logger.info(f"Transaction cost: {transaction_cost:.6f}")
            logger.info(f"Net return: {net_return:.6f}")
            logger.info(f"Portfolio value change: {initial_capital:.2f} -> {portfolio_value:.2f}")
        
        step += 1
        
        # Progress log (at 20% intervals)
        if step % (env.max_steps // 5) == 0:
            logger.info(f"Simulation progress: {step}/{env.max_steps} ({step/env.max_steps*100:.1f}%)")
    
    logger.info(f"Simulation completed: ran for {step} days")
    
    # Output debugging information
    if debug:
        logger.info("\n===== Debugging Information =====")
        logger.info(f"First 10 daily returns: {returns[:10]}")
        logger.info(f"First 10 portfolio values: {portfolio_values[:10]}")
        logger.info(f"First 10 volatilities: {vols[:10]}")
        logger.info(f"First 10 weights: {weights_history[:10]}")
        
        # Return statistics
        logger.info(f"\nReturn Statistics:")
        logger.info(f"Min: {min(returns):.4f}%, Max: {max(returns):.4f}%, Mean: {np.mean(returns):.4f}%")
        logger.info(f"Standard Deviation: {np.std(returns):.4f}%")
        
        # Check for outliers
        outliers = [r for r in returns if abs(r) > 5.0]  # Consider daily returns over 5% as outliers
        if outliers:
            logger.info(f"\nOutlier returns ({len(outliers)}): {outliers}")
        
        # Check data scale
        logger.info("\n===== Data Scale Check =====")
        stock_returns_scale = []
        for i in range(10):  # Check return range for each of 10 stocks
            returns_idx = i * 4  # Return is the first feature
            stock_returns = [env.market_data[j, returns_idx] for j in range(len(env.market_data))]
            logger.info(f"Stock {i+1} return range: {min(stock_returns):.4f} ~ {max(stock_returns):.4f}, Mean: {np.mean(stock_returns):.4f}")
            stock_returns_scale.append((min(stock_returns), max(stock_returns), np.mean(stock_returns)))
        
        # Output sample of original market data
        logger.info("\nOriginal market data sample (first 3 days):")
        for day in range(min(3, len(env.market_data))):
            logger.info(f"Day {day+1}: {env.market_data[day, :10]}")  # Only output first 10 values
    
    # Create results dataframe
    results = pd.DataFrame({
        'Date': dates,
        'Portfolio_Value': portfolio_values,
        'Daily_Return': returns,
        'Volatility': vols
    })
    
    # Calculate cumulative return
    results['Cumulative_Return'] = (results['Portfolio_Value'] / initial_capital - 1) * 100
    
    # Save daily return log to CSV file
    daily_returns_df = pd.DataFrame({
        'Date': dates,
        'Daily_Return': returns,
        'Net_Return': [r - 0.001 * t/100.0 for r, t in zip(returns, turnovers)],
        'Portfolio_Value': portfolio_values
    })
    daily_returns_df.to_csv('daily_returns_log.csv', index=False)
    logger.info(f"\nDaily returns log has been saved to 'daily_returns_log.csv'")
    
    # Calculate performance metrics
    total_return = (portfolio_values[-1] / initial_capital - 1) * 100
    annualized_return = ((1 + total_return/100) ** (252/len(portfolio_values)) - 1) * 100
    
    # Calculate net returns (after transaction costs)
    net_returns = [r - 0.001 * t/100.0 for r, t in zip(returns, turnovers)]
    
    # Calculate annualized volatility - based on net returns
    net_returns_array = np.array(net_returns) * 100  # Convert to percentage
    daily_std = np.std(net_returns_array)
    annualized_vol = daily_std * np.sqrt(252)
    
    # Calculate risk-free rate
    # Use a fixed value for consistency in evaluation
    # In a real problem, market data should be used,
    # but a constant value is used for comparative evaluation
    annual_risk_free_rate = 2.0  # Assume 2% annual rate
    
    # Calculate Sharpe ratio: (annual return - risk-free rate) / annual volatility
    sharpe_ratio = (annualized_return - annual_risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
    
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # Output results
    logger.info(f"\n===== Model Evaluation Results =====")
    logger.info(f"Seed: {seed}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Annualized Return: {annualized_return:.2f}%")
    logger.info(f"Average Daily Net Return: {np.mean(net_returns) * 100:.4f}%")
    logger.info(f"Annualized Volatility: {annualized_vol:.2f}%")
    logger.info(f"Risk-Free Rate: {annual_risk_free_rate:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Visualize portfolio value trend
    logger.info("Creating portfolio performance chart...")
    plt.figure(figsize=(14, 10))
    
    # Portfolio value chart
    plt.subplot(2, 1, 1)
    plt.plot(results['Date'], results['Portfolio_Value'], 'b-', linewidth=2)
    plt.title('Portfolio Value Over Time', fontsize=15)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Daily returns chart (changed to net returns)
    plt.subplot(2, 1, 2)
    
    # Show both original returns and net returns
    plt.plot(results['Date'], returns, 'g-', alpha=0.5, linewidth=1, label='Gross Returns')
    plt.plot(results['Date'], net_returns, 'r-', linewidth=1, label='Net Returns (after costs)')
    
    plt.title('Daily Returns', fontsize=15)
    plt.ylabel('Return (%)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    logger.info("Portfolio performance chart has been saved to 'portfolio_performance.png'")
    
    # Visualize stock weight trends
    logger.info("Creating portfolio weights chart...")
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
    logger.info("Portfolio weights chart has been saved to 'portfolio_weights.png'")
    logger.info("Model evaluation completed")
    
    return results, weights_array

def calculate_max_drawdown(portfolio_values):
    """Calculate Maximum Drawdown"""
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
    """Compare the performance of multiple models."""
    logger.info(f"Starting model comparison: {model_paths}")
    all_results = []
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        logger.info(f"\n----- Evaluating Model: {model_name} -----")
        
        # Evaluate across multiple seeds
        seed_results = []
        for seed in seeds:
            logger.info(f"Evaluating with seed {seed}...")
            env = PortfolioEnv(seed=seed)
            model = PPO.load(model_path)
            
            obs, _ = env.reset()
            done = False
            
            portfolio_value = initial_capital
            portfolio_values = [portfolio_value]
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                
                # Normalize action: same as in the environment's step method
                action = action - np.mean(action)
                action = action / (np.sum(np.abs(action)) + 1e-8)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Calculate daily return - convert to decimal
                daily_return = info["portfolio_return"]  # Already in decimal form (e.g., 0.01 = 1%)
                turnover = info["turnover"]
                
                # Calculate transaction cost
                transaction_cost = 0.001 * turnover
                
                # Apply net return
                net_return = daily_return - transaction_cost
                portfolio_value *= (1 + net_return)
                
                portfolio_values.append(portfolio_value)
            
            # Calculate performance metrics
            total_return = (portfolio_values[-1] / initial_capital - 1) * 100
            seed_results.append(total_return)
            logger.info(f"Seed {seed} evaluation result: Return {total_return:.2f}%")
        
        # Calculate average and standard deviation
        avg_return = np.mean(seed_results)
        std_return = np.std(seed_results)
        
        all_results.append({
            'Model': model_name,
            'Avg_Return': avg_return,
            'Std_Return': std_return,
            'Min_Return': min(seed_results),
            'Max_Return': max(seed_results)
        })
        
        logger.info(f"Model {model_name} evaluation completed:")
        logger.info(f"  Average Return: {avg_return:.2f}%")
        logger.info(f"  Standard Deviation: {std_return:.2f}%")
        logger.info(f"  Min/Max: {min(seed_results):.2f}% / {max(seed_results):.2f}%")
    
    # Create and output results dataframe
    results_df = pd.DataFrame(all_results)
    logger.info("\n===== Model Comparison Results =====")
    logger.info(f"\n{results_df}")
    
    # Visualize results
    logger.info("Creating model comparison chart...")
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['Model'], results_df['Avg_Return'], yerr=results_df['Std_Return'], 
            capsize=5, color='skyblue', alpha=0.7)
    plt.title('Average Return by Model', fontsize=15)
    plt.ylabel('Average Return (%)', fontsize=12)
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    logger.info("Model comparison chart has been saved to 'model_comparison.png'")
    
    return results_df

def robust_evaluation(model_path, seeds=range(1000, 1100), initial_capital=10000):
    """Evaluate model robustness across various seeds."""
    logger.info(f"Starting robustness evaluation: {model_path}, Test seeds: {len(seeds)}")
    results = []
    
    for i, seed in enumerate(seeds):
        try:
            # Log progress
            if i % 10 == 0:
                logger.info(f"Robustness evaluation progress: {i}/{len(seeds)} ({i/len(seeds)*100:.1f}%)")
                
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
                
                daily_return = info["portfolio_return"]  # Already in decimal form (e.g., 0.01 = 1%)
                turnover = info["turnover"]
                transaction_cost = 0.001 * turnover
                
                net_return = daily_return - transaction_cost
                portfolio_value *= (1 + net_return)
            
            total_return = (portfolio_value / initial_capital - 1) * 100
            results.append(total_return)
            
            # Log brief results every 20 seeds
            if i % 20 == 19:
                logger.info(f"Seed {seed}: Return {total_return:.2f}%")
        
        except Exception as e:
            logger.error(f"Error during seed {seed} evaluation: {e}")
            continue
    
    # Summarize results
    if results:
        logger.info(f"\n===== Robustness Evaluation Results ({len(results)} seeds) =====")
        logger.info(f"Average Return: {np.mean(results):.2f}%")
        logger.info(f"Return Standard Deviation: {np.std(results):.2f}%")
        logger.info(f"Minimum Return: {min(results):.2f}%")
        logger.info(f"Maximum Return: {max(results):.2f}%")
        
        # Visualize with histogram
        logger.info("Creating return distribution histogram...")
        plt.figure(figsize=(10, 6))
        plt.hist(results, bins=20, alpha=0.7)
        plt.title('Return Distribution Across Different Seeds')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('return_distribution.png')
        logger.info("Return distribution histogram has been saved to 'return_distribution.png'")
    else:
        logger.warning("No evaluation results available.")
    
    return results

def start_evaluation_server():
    """Start evaluation server"""
    app = start_server()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    try:
        logger.info("="*50)
        logger.info("Starting Portfolio Optimization Model Evaluation")
        logger.info("="*50)
        
        # Run FastAPI web server in a separate thread
        server_thread = threading.Thread(target=start_evaluation_server)
        server_thread.daemon = True  # Terminate with main thread
        server_thread.start()
        
        logger.info("Web interface is running at http://localhost:8000")
        time.sleep(1)  # Give server time to start
        
        # Evaluate single model
        model_path = "ppo_portfolio_best"  # or "ppo_portfolio_best"
        results, weights = evaluate_model(model_path, seed=1234)
        
        # Compare multiple models (optional)
        # model_paths = ["ppo_portfolio", "ppo_portfolio_best", "save/ppo_portfolio_best_v1"]
        # compare_results = compare_models(model_paths)
        
        # Robustness evaluation (optional)
        # robust_results = robust_evaluation(model_path, seeds=range(1000, 1050))
        
        logger.info("Model evaluation completed")
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        logger.error(traceback.format_exc())
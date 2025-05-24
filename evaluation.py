import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend to avoid conflicts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO
from env import PortfolioEnv
from models import NormalizedActorCriticPolicy
import os
import seaborn as sns
from datetime import datetime
import logging
import traceback
import time
import fcntl  # For file locking on Unix systems
import platform
from app.utils import setup_logger

# Get logger
logger = setup_logger()

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def safe_model_load(model_path, max_retries=5, retry_delay=2):
    """
    Safely load model with retry mechanism to avoid conflicts with training
    """
    for attempt in range(max_retries):
        try:
            # Check if model file exists and is not being written to
            model_file = f"{model_path}.zip"
            if not os.path.exists(model_file):
                logger.warning(f"Model file not found: {model_file}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise FileNotFoundError(f"Model file not found after {max_retries} attempts: {model_file}")
            
            # Try to acquire file lock (Unix systems only)
            if platform.system() != 'Windows':
                try:
                    with open(model_file, 'rb') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # File is available, release lock and proceed
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except IOError:
                    logger.warning(f"Model file is locked (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error("Could not acquire file lock after maximum retries")
                        raise
            
            # Load model
            logger.info(f"Loading model from: {model_file}")
            model = PPO.load(model_path, custom_objects={"policy_class": NormalizedActorCriticPolicy})
            logger.info(f"Model loaded successfully: {model_path}")
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load model (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to load model after {max_retries} attempts")
                raise

def evaluate_model(model_path, seed=None, initial_capital=10000, debug=False):
    """
    Evaluates the trained model and visualizes portfolio value trends.
    
    Args:
        model_path: Path to the trained model
        seed: Seed to use for evaluation
        initial_capital: Initial investment amount
        debug: Whether to output debugging information
    """
    try:
        if seed is None:
            seed = int(time.time()) % 10000
            
        logger.info(f"Starting model evaluation: {model_path} (seed: {seed})")
        
        # Create environment
        env = PortfolioEnv(seed=seed)
        
        # Safely load model with retry mechanism
        model = safe_model_load(model_path, max_retries=3, retry_delay=1)
        
        # Run evaluation
        obs, _ = env.reset()
        done = False
        
        # Lists for storing results
        dates = []
        portfolio_values = []
        returns = []
        vols = []
        weights_history = []
        turnovers = []
        
        # Create date data (arbitrary creation since scenario doesn't have date information)
        start_date = datetime(2022, 1, 1)
        
        # Run episode
        step = 0
        portfolio_value = initial_capital
        
        logger.info("Starting portfolio simulation...")
        while not done:
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as e:
                action = np.zeros(env.action_space.shape)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                logger.warning(f"Error during evaluation step {step}: {e}")
                continue
            
            # Add date
            current_date = start_date + pd.Timedelta(days=step)
            dates.append(current_date)
            
            # Calculate and add portfolio value
            daily_return = info["portfolio_return"]
            turnover = info["turnover"]
            turnovers.append(turnover)
            
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
            weights_history.append(env.previous_action.copy())
            
            step += 1
            
            # Progress log (at 20% intervals) - less verbose
            if step % (env.max_steps // 5) == 0:
                logger.info(f"Simulation progress: {step}/{env.max_steps} ({step/env.max_steps*100:.1f}%)")
        
        logger.info(f"Portfolio simulation completed: {step} days")
        
        # Calculate performance metrics
        total_return = (portfolio_values[-1] / initial_capital - 1) * 100
        annualized_return = ((1 + total_return/100) ** (252/len(portfolio_values)) - 1) * 100
        
        # Calculate net returns (after transaction costs)
        net_returns = [r - 0.001 * t for r, t in zip(returns, turnovers)]
        
        # Calculate annualized volatility - based on net returns
        net_returns_array = np.array(net_returns) * 100
        daily_std = np.std(net_returns_array)
        annualized_vol = daily_std * np.sqrt(252)
        
        # Calculate Sharpe ratio
        annual_risk_free_rate = 2.0
        sharpe_ratio = (annualized_return - annual_risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
        
        max_drawdown = calculate_max_drawdown(portfolio_values)
        
        # Output results
        logger.info(f"\n===== Model Evaluation Results =====")
        logger.info(f"Seed: {seed}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Annualized Return: {annualized_return:.2f}%")
        logger.info(f"Annualized Volatility: {annualized_vol:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")
        
        # Generate charts
        logger.info("Generating portfolio performance charts...")
        create_performance_charts(dates, portfolio_values, returns, net_returns, weights_history)
        logger.info("Charts generated successfully")
        
        # Create results dataframe
        results = pd.DataFrame({
            'Date': dates,
            'Portfolio_Value': portfolio_values,
            'Daily_Return': returns,
            'Volatility': vols
        })
        
        # Calculate cumulative return
        results['Cumulative_Return'] = (results['Portfolio_Value'] / initial_capital - 1) * 100
        
        logger.info("Model evaluation completed successfully")
        return results, np.array(weights_history)
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_performance_charts(dates, portfolio_values, returns, net_returns, weights_history):
    """
    Create performance charts without conflicting with training
    """
    try:
        # Portfolio performance chart
        plt.figure(figsize=(14, 10))
        
        # Portfolio value chart
        plt.subplot(2, 1, 1)
        plt.plot(dates, portfolio_values, 'b-', linewidth=2)
        plt.title('Portfolio Value Over Time', fontsize=15)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Daily returns chart
        plt.subplot(2, 1, 2)
        plt.plot(dates, [r * 100 for r in returns], 'g-', alpha=0.7, linewidth=1, label='Gross Returns')
        plt.plot(dates, [r * 100 for r in net_returns], 'r-', linewidth=1, label='Net Returns (after costs)')
        
        plt.title('Daily Returns', fontsize=15)
        plt.ylabel('Return (%)', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('portfolio_performance.png', dpi=100, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        # Portfolio weights chart
        plt.figure(figsize=(14, 8))
        weights_array = np.array(weights_history)
        
        for i in range(weights_array.shape[1]):
            plt.plot(dates, weights_array[:, i], label=f'Stock {i+1}', alpha=0.8)
        
        plt.title('Portfolio Weights Over Time', fontsize=15)
        plt.ylabel('Weight', fontsize=12)
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('portfolio_weights.png', dpi=100, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
    except Exception as e:
        logger.error(f"Error creating charts: {str(e)}")
        raise

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

# REMOVED: Server startup and main execution
# The evaluation should only provide the evaluation function
# and not start its own server or conflict with training

if __name__ == "__main__":
    # Only for standalone testing - not used by the web interface
    logger.info("Running standalone evaluation (testing mode)")
    try:
        model_path = "ppo_portfolio"
        if os.path.exists(f"{model_path}.zip"):
            results, weights = evaluate_model(model_path, seed=1234)
            logger.info("Standalone evaluation completed")
        else:
            logger.warning(f"Model file not found: {model_path}.zip")
    except Exception as e:
        logger.error(f"Standalone evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())
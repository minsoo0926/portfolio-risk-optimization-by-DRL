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

def evaluate_model(model_path, seed=None, initial_capital=10000, debug=False, log_weights=True):
    """
    Evaluates the trained model and visualizes portfolio value trends.
    
    Args:
        model_path: Path to the trained model
        seed: Seed to use for evaluation
        initial_capital: Initial investment amount
        debug: Whether to output debugging information
        log_weights: Whether to log detailed weights history
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
        
        # Initialize weights logging
        if log_weights:
            logger.info("="*60)
            logger.info("WEIGHTS HISTORY LOGGING ENABLED")
            logger.info("="*60)
            weights_log_file = f"weights_history_{seed}_{int(time.time())}.txt"
            
        while not done:
            try:
                action, _ = model.predict(obs, deterministic=False)
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
            
            # Store and log weights
            current_weights = env.previous_action.copy()
            weights_history.append(current_weights)
            
            # Detailed weights logging
            if log_weights:
                # Log every 10 steps to avoid overwhelming output
                if step % 10 == 0 or step < 5:
                    logger.info(f"Step {step:3d}: Weights = [{', '.join([f'{w:6.3f}' for w in current_weights])}]")
                    logger.info(f"         Return = {daily_return:7.4f}, Vol = {info['portfolio_vol']:6.4f}, Turnover = {turnover:6.4f}")
                
                # Log weights statistics every 50 steps
                if step % 50 == 0 and step > 0:
                    weights_array = np.array(weights_history)
                    logger.info(f"\n--- Weights Statistics (Steps 0-{step}) ---")
                    logger.info(f"Mean weights:     [{', '.join([f'{w:6.3f}' for w in np.mean(weights_array, axis=0)])}]")
                    logger.info(f"Std weights:      [{', '.join([f'{w:6.3f}' for w in np.std(weights_array, axis=0)])}]")
                    logger.info(f"Max weights:      [{', '.join([f'{w:6.3f}' for w in np.max(weights_array, axis=0)])}]")
                    logger.info(f"Min weights:      [{', '.join([f'{w:6.3f}' for w in np.min(weights_array, axis=0)])}]")
                    
                    # Log concentration metrics
                    avg_weights = np.mean(weights_array, axis=0)
                    concentration = np.sum(avg_weights ** 2)  # Herfindahl index
                    max_weight = np.max(np.abs(avg_weights))
                    logger.info(f"Concentration:    {concentration:.4f} (lower = more diversified)")
                    logger.info(f"Max weight:       {max_weight:.4f}")
                    logger.info("-" * 50)
            
            step += 1
            
            # Progress log (at 20% intervals) - less verbose
            if step % (env.max_steps // 5) == 0:
                logger.info(f"Simulation progress: {step}/{env.max_steps} ({step/env.max_steps*100:.1f}%)")
        
        logger.info(f"Portfolio simulation completed: {step} days")
        
        # Final weights analysis
        if log_weights and weights_history:
            logger.info("\n" + "="*60)
            logger.info("FINAL WEIGHTS ANALYSIS")
            logger.info("="*60)
            
            weights_array = np.array(weights_history)
            
            # Overall statistics
            logger.info(f"Total trading days: {len(weights_history)}")
            logger.info(f"Final weights:    [{', '.join([f'{w:6.3f}' for w in weights_history[-1]])}]")
            logger.info(f"Average weights:  [{', '.join([f'{w:6.3f}' for w in np.mean(weights_array, axis=0)])}]")
            logger.info(f"Weight volatility:[{', '.join([f'{w:6.3f}' for w in np.std(weights_array, axis=0)])}]")
            
            # Portfolio characteristics
            avg_concentration = np.mean([np.sum(w**2) for w in weights_array])
            avg_turnover = np.mean(turnovers)
            max_single_weight = np.max(np.abs(weights_array))
            
            logger.info(f"\nPortfolio Characteristics:")
            logger.info(f"Average concentration: {avg_concentration:.4f}")
            logger.info(f"Average turnover:      {avg_turnover:.4f}")
            logger.info(f"Maximum single weight: {max_single_weight:.4f}")
            
            # Long/short analysis
            long_positions = weights_array > 0
            short_positions = weights_array < 0
            
            avg_long_weight = np.mean(weights_array[long_positions]) if np.any(long_positions) else 0
            avg_short_weight = np.mean(weights_array[short_positions]) if np.any(short_positions) else 0
            long_ratio = np.mean(np.sum(long_positions, axis=1)) / weights_array.shape[1]
            
            logger.info(f"\nLong/Short Analysis:")
            logger.info(f"Average long weight:   {avg_long_weight:.4f}")
            logger.info(f"Average short weight:  {avg_short_weight:.4f}")
            logger.info(f"Long positions ratio:  {long_ratio:.4f} ({long_ratio*100:.1f}%)")
            
            # Save weights history to file
            try:
                with open(weights_log_file, 'w') as f:
                    f.write("# Portfolio Weights History\n")
                    f.write(f"# Model: {model_path}, Seed: {seed}\n")
                    f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("# Columns: Step, Date, Stock1, Stock2, ..., Stock10, Return, Volatility, Turnover\n")
                    
                    for i, (date, weights, ret, vol, turn) in enumerate(zip(dates, weights_history, returns, vols, turnovers)):
                        f.write(f"{i:3d}, {date.strftime('%Y-%m-%d')}, ")
                        f.write(", ".join([f"{w:8.5f}" for w in weights]))
                        f.write(f", {ret:8.5f}, {vol:8.5f}, {turn:8.5f}\n")
                
                logger.info(f"Weights history saved to: {weights_log_file}")
            
            except Exception as e:
                logger.warning(f"Failed to save weights history: {e}")
        
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
        # Portfolio performance chart - INCREASED FIGURE SIZE AND IMPROVED LAYOUT
        plt.figure(figsize=(18, 14))  # Larger figure size
        
        # Portfolio value chart
        plt.subplot(3, 2, 1)
        plt.plot(dates, portfolio_values, 'b-', linewidth=2)
        plt.title('Portfolio Value Over Time', fontsize=14)  # Reduced font size
        plt.ylabel('Portfolio Value ($)', fontsize=11)
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Shorter date format
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45, fontsize=9)
        
        # Daily returns chart
        plt.subplot(3, 2, 2)
        plt.plot(dates, [r * 100 for r in returns], 'g-', alpha=0.7, linewidth=1, label='Gross Returns')
        plt.plot(dates, [r * 100 for r in net_returns], 'r-', linewidth=1, label='Net Returns (after costs)')
        
        plt.title('Daily Returns', fontsize=14)
        plt.ylabel('Return (%)', fontsize=11)
        plt.grid(True)
        plt.legend(fontsize=9)  # Smaller legend
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45, fontsize=9)
        
        # Portfolio weights over time - SIMPLIFIED LEGEND
        plt.subplot(3, 2, 3)
        weights_array = np.array(weights_history)
        
        # Only show first 5 stocks to avoid legend clutter
        for i in range(min(5, weights_array.shape[1])):
            plt.plot(dates, weights_array[:, i], label=f'S{i+1}', alpha=0.8)
        
        plt.title('Portfolio Weights (Top 5)', fontsize=14)
        plt.ylabel('Weight', fontsize=11)
        plt.grid(True)
        plt.legend(fontsize=8, loc='upper right')  # Smaller legend, better position
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45, fontsize=9)
        
        # Portfolio concentration over time
        plt.subplot(3, 2, 4)
        concentration = [np.sum(w**2) for w in weights_array]
        plt.plot(dates, concentration, 'purple', linewidth=2)
        plt.title('Portfolio Concentration', fontsize=14)
        plt.ylabel('Concentration', fontsize=11)
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45, fontsize=9)
        
        # Average weights distribution
        plt.subplot(3, 2, 5)
        avg_weights = np.mean(weights_array, axis=0)
        stock_labels = [f'S{i+1}' for i in range(len(avg_weights))]
        colors = ['red' if w < 0 else 'green' for w in avg_weights]
        
        bars = plt.bar(stock_labels, avg_weights, color=colors, alpha=0.7)
        plt.title('Average Portfolio Weights', fontsize=14)
        plt.ylabel('Average Weight', fontsize=11)
        plt.grid(True, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xticks(fontsize=9)
        
        # Add value labels on bars - SMALLER TEXT
        for bar, weight in zip(bars, avg_weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if weight > 0 else -0.01),
                    f'{weight:.2f}', ha='center', va='bottom' if weight > 0 else 'top', fontsize=8)
        
        # Weights volatility
        plt.subplot(3, 2, 6)
        weight_stds = np.std(weights_array, axis=0)
        plt.bar(stock_labels, weight_stds, color='orange', alpha=0.7)
        plt.title('Portfolio Weights Volatility', fontsize=14)
        plt.ylabel('Standard Deviation', fontsize=11)
        plt.grid(True, axis='y')
        plt.xticks(fontsize=9)
        
        # Add value labels on bars - SMALLER TEXT
        for i, std in enumerate(weight_stds):
            plt.text(i, std + 0.001, f'{std:.3f}', ha='center', va='bottom', fontsize=8)
        
        # IMPROVED LAYOUT WITH MANUAL SPACING (avoids tight_layout warnings)
        plt.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.93, wspace=0.3, hspace=0.5)
        
        plt.savefig('portfolio_performance.png', dpi=100, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        # Separate detailed weights chart - IMPROVED LAYOUT
        plt.figure(figsize=(18, 12))  # Larger figure
        
        # Individual stock weights over time (larger view)
        plt.subplot(2, 1, 1)
        for i in range(weights_array.shape[1]):
            plt.plot(dates, weights_array[:, i], label=f'Stock {i+1}', linewidth=2, alpha=0.8)
        
        plt.title('Detailed Portfolio Weights Over Time', fontsize=16)
        plt.ylabel('Weight', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)  # Move legend outside
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.xticks(rotation=45, fontsize=10)
        
        # Portfolio statistics over time
        plt.subplot(2, 1, 2)
        
        # Calculate rolling statistics
        window = min(20, len(weights_array) // 10)  # 20-day or 10% of data window
        if window >= 5:
            rolling_concentration = []
            rolling_max_weight = []
            rolling_turnover = []
            
            for i in range(window-1, len(weights_array)):
                window_weights = weights_array[max(0, i-window+1):i+1]
                rolling_concentration.append(np.mean([np.sum(w**2) for w in window_weights]))
                rolling_max_weight.append(np.mean([np.max(np.abs(w)) for w in window_weights]))
                
                if i > 0:
                    window_turnover = [np.sum(np.abs(weights_array[j] - weights_array[j-1])) 
                                     for j in range(max(1, i-window+1), i+1)]
                    rolling_turnover.append(np.mean(window_turnover))
                else:
                    rolling_turnover.append(0)
            
            dates_subset = dates[window-1:]
            
            plt.plot(dates_subset, rolling_concentration, 'purple', linewidth=2, label=f'Concentration ({window}d avg)')
            plt.plot(dates_subset, rolling_max_weight, 'orange', linewidth=2, label=f'Max Weight ({window}d avg)')
            plt.plot(dates_subset, rolling_turnover, 'red', linewidth=2, label=f'Turnover ({window}d avg)')
        else:
            # Fallback for small datasets
            plt.plot(dates, concentration, 'purple', linewidth=2, label='Concentration')
            plt.plot(dates, [np.max(np.abs(w)) for w in weights_array], 'orange', linewidth=2, label='Max Weight')
        
        plt.title('Portfolio Risk Metrics Over Time', fontsize=16)
        plt.ylabel('Metric Value', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.xticks(rotation=45, fontsize=10)
        
        # IMPROVED LAYOUT WITH PROPER SPACING (avoids tight_layout warnings)
        plt.subplots_adjust(left=0.08, bottom=0.12, right=0.85, top=0.93, hspace=0.35)
        
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
            logger.info("Starting evaluation with weights history logging enabled")
            results, weights = evaluate_model(model_path, seed=1234, log_weights=True)
            logger.info("Standalone evaluation completed")
            logger.info("Check the generated weights_history_*.txt file for detailed weights data")
        else:
            logger.warning(f"Model file not found: {model_path}.zip")
    except Exception as e:
        logger.error(f"Standalone evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())
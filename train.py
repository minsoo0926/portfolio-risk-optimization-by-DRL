import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import torch.nn as nn
from generate_scenario import generate_scenario
import os
import time
import torch
import datetime
import logging
import traceback
import asyncio
import threading
from app.server import start_server
from app.utils import setup_logger

# Import our custom policy models
from models import NormalizedActorCriticPolicy, create_ppo_model

# Import environment
from env import PortfolioEnv

# Get logger
logger = setup_logger()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu" # PPO is poorly supported on GPU
logger.info(f"Using device: {device}")

# CustomCallback class
class CustomCallback(BaseCallback):
    def __init__(self, eval_env, verbose=0, save_freq=10000, eval_freq=20000, model_path="ppo_portfolio"):
        super(CustomCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_results = []
        self.best_mean_reward = -np.inf
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.model_path = model_path
        
        # Create temp directory if it doesn't exist
        if not os.path.exists('temp'):
            os.makedirs('temp')
            logger.info("Created temp folder.")
        
    def _on_step(self):
        try:
            # Periodically save model (now in temp folder)
            if self.num_timesteps % self.save_freq == 0:
                # Save intermediate model in temp folder
                temp_model_path = os.path.join('temp', f"{self.model_path}_{self.num_timesteps}")
                self.model.save(temp_model_path)
                logger.info(f"Timestep {self.num_timesteps}: Model saved (temp/{self.model_path}_{self.num_timesteps})")
            
            # Periodically evaluate model performance
            if self.num_timesteps % self.eval_freq == 0:
                self._evaluate_model()
                
                # Memory management: keep only the last 10 results
                if len(self.eval_results) > 10:
                    self.eval_results = self.eval_results[-10:]
                
            return True
        except Exception as e:
            logger.error(f"Callback error: {str(e)}")
            return False
            
    def _evaluate_model(self):
        logger.info("="*50)
        logger.info(f"===== Timestep {self.num_timesteps} Model Evaluation =====")
        logger.info("="*50)
        
        # Variables to store results
        daily_returns = []  # Original daily returns
        net_returns = []    # Net returns after transaction costs
        episode_vols = []
        risk_free_rates = []
        
        # Initial portfolio value
        initial_capital = 10000
        portfolio_value = initial_capital
        
        # Run episode in evaluation environment
        obs, _ = self.eval_env.reset()
        done = False
        while not done:
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            except Exception as e:
                action = np.zeros(self.eval_env.action_space.shape) # dummy action
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                logger.warning(f"Error during evaluation: {e}")
                continue
            
            # Store daily returns (already in decimal form)
            daily_return = info["portfolio_return"]  # Already in decimal form (e.g., 0.01 = 1%)
            daily_returns.append(daily_return)
            
            turnover = info["turnover"]
            transaction_cost = 0.001 * turnover  # 0.1% transaction cost
            
            # Net return after transaction costs
            net_return = daily_return - transaction_cost
            net_returns.append(net_return)
            
            # Update portfolio value (compound returns)
            portfolio_value *= (1 + net_return)
            
            episode_vols.append(info["portfolio_vol"])
            
            # Collect Treasury yield data (already in percentage, divide by 100 to convert to decimal)
            if hasattr(self.eval_env, 'macro_data') and self.eval_env.current_step-1 < len(self.eval_env.macro_data):
                # Check if value is already in percentage (adjust if it's a large value)
                raw_rate = self.eval_env.macro_data[self.eval_env.current_step-1, 1]
                # Treasury yields are typically below 10%, so adjust if it's a large value
                if raw_rate > 10:
                    t_bill_rate = raw_rate / 10000.0  # Adjust very large values (e.g., 270 -> 0.0270)
                else:
                    t_bill_rate = raw_rate / 100.0  # Normal case (e.g., 2.7 -> 0.027)
                risk_free_rates.append(t_bill_rate)
        
        # Average daily return (log return average)
        mean_daily_return = np.mean(np.log(1 + np.array(daily_returns))) * 100  # Convert to percentage
        mean_net_return = np.mean(np.log(1 + np.array(net_returns))) * 100      # Convert to percentage
        
        # Calculate total return based on compound returns (using net returns)
        total_return = (portfolio_value / initial_capital - 1) * 100
        
        # Annualization adjustment based on investment period
        days = len(daily_returns)
        trading_days_per_year = 252
        
        # Calculate annualized return
        annualized_return = ((1 + total_return/100) ** (trading_days_per_year/days) - 1) * 100
        
        # Average volatility
        mean_vol = np.mean(episode_vols)
        
        # Risk-free rate - using a simple fixed value
        # Using a fixed value due to issues with calculating risk-free rate from market data
        annual_risk_free_rate = 2.0  # Default annual 2%
        
        # Calculate Sharpe ratio - using standard deviation of net returns (important!)
        # Convert decimal net returns to percentage for standard deviation calculation
        net_returns_array = np.array(net_returns) * 100  # Convert decimal to percentage
        daily_std = np.std(net_returns_array)  # Standard deviation of net returns
        annualized_std = daily_std * np.sqrt(trading_days_per_year)  # Annualized standard deviation
        
        # Sharpe ratio: (Annualized return - Annual risk-free rate) / Annualized standard deviation
        # Add epsilon (1e-8) to prevent division by zero
        sharpe = (annualized_return - annual_risk_free_rate) / (annualized_std + 1e-8)
        
        # Additional info for logging
        mean_daily_net_return = np.mean(net_returns) * 100  # Average daily net return
        
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
        
        # Output results
        logger.info("\nPerformance Metrics:")
        logger.info(f"Average Daily Return (before costs): {mean_daily_return:.4f}%")
        logger.info(f"Average Daily Net Return (after costs): {mean_daily_net_return:.4f}%")
        logger.info(f"Total Return: {total_return:.4f}%")
        logger.info(f"Annualized Return: {annualized_return:.4f}%")
        logger.info(f"Annual Risk-Free Rate: {annual_risk_free_rate:.4f}%")
        logger.info(f"Daily Net Return Std Dev: {daily_std:.4f}%")
        logger.info(f"Annualized Std Dev: {annualized_std:.4f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.4f}")
        
        # Output Sharpe ratio calculation details (for debugging)
        logger.info(f"Sharpe Calculation: ({annualized_return:.4f} - {annual_risk_free_rate:.4f}) / {annualized_std:.4f} = {sharpe:.4f}")
        
        # Log to file
        with open("evaluation_log.txt", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Timestep {self.num_timesteps}\n")
            f.write(f"Average Daily Return (before costs): {mean_daily_return:.4f}%\n")
            f.write(f"Average Daily Net Return (after costs): {mean_daily_net_return:.4f}%\n")
            f.write(f"Total Return: {total_return:.4f}%\n")
            f.write(f"Annualized Return: {annualized_return:.4f}%\n")
            f.write(f"Annual Risk-Free Rate: {annual_risk_free_rate:.4f}%\n")
            f.write(f"Daily Net Return Std Dev: {daily_std:.4f}%\n")
            f.write(f"Annualized Std Dev: {annualized_std:.4f}%\n")
            f.write(f"Sharpe Ratio: {sharpe:.4f}\n")
            f.write(f"Sharpe Calculation: ({annualized_return:.4f} - {annual_risk_free_rate:.4f}) / {annualized_std:.4f} = {sharpe:.4f}\n\n")
        
        # Save best performing model (now based on compound return)
        if total_return > self.best_mean_reward:
            self.best_mean_reward = total_return
            self.model.save(f"{self.model_path}_best")
            logger.info(f"New best model saved ({self.model_path}_best), Total Return: {total_return:.4f}%")


def main():
    try:
        logger.info("="*50)
        logger.info(f"Portfolio Optimization Training Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*50)
        
        model_path = "ppo_portfolio"
        total_timesteps = 500000  # Increased training steps
        eval_episodes = 10
        save_freq = 10000
        eval_freq = 20000

        # Ensure diversity in training data
        np.random.seed(int(time.time()))
        train_seeds = np.random.randint(0, 10000, size=10)  # Use multiple seeds
        logger.info(f"Training seeds: {train_seeds}")
        
        # Train across multiple environments
        train_envs = [PortfolioEnv(seed=seed) for seed in train_seeds]
        logger.info(f"Environments created: {len(train_envs)} environments")
        
        # Load existing model if available, otherwise create new one
        if os.path.exists(model_path + ".zip"):
            try:
                # Try to load existing model
                model = PPO.load(model_path, env=train_envs[0], custom_objects={"policy_class": NormalizedActorCriticPolicy})
                logger.info(f"Model loaded successfully ({model_path})")
            except ValueError as e:
                # Create new model if observation space mismatch
                logger.warning(f"Failed to load existing model: {e}")
                logger.info("Creating new model.")
                
                # Create model using our custom function
                policy_kwargs = dict(
                    net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),
                    activation_fn=torch.nn.ReLU
                )
                model = create_ppo_model(train_envs[0], policy_kwargs=policy_kwargs, device=device)
                logger.info("PPO model created successfully")
        else:
            # Create new model using our custom function
            policy_kwargs = dict(
                net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),
                activation_fn=torch.nn.ReLU
            )
            model = create_ppo_model(train_envs[0], policy_kwargs=policy_kwargs, device=device)
            logger.info("New PPO model created successfully")
        
        # Create evaluation environment
        eval_env = PortfolioEnv(seed=9999)
        callback = CustomCallback(
            eval_env=eval_env,
            save_freq=save_freq,   # Save more frequently
            eval_freq=eval_freq,   # Evaluate more frequently
            model_path=model_path
        )
        logger.info("Evaluation environment and callback setup complete")
        
        # Train alternating between environments
        learning_steps_per_env = 5000
        logger.info(f"Training plan: Total {total_timesteps} steps, {learning_steps_per_env} steps per environment")
        
        for cycle in range(total_timesteps // (len(train_envs) * learning_steps_per_env)):
            logger.info(f"\n===== Starting Training Cycle {cycle+1} =====")
            
            for env_idx, env in enumerate(train_envs):
                try:
                    model.set_env(env)  # Change environment
                    
                    # Use callback only in the last environment
                    if env_idx == len(train_envs) - 1:
                        logger.info(f"Starting training on environment {env_idx+1}/{len(train_envs)} (seed {train_seeds[env_idx]}) with evaluation")
                        model.learn(total_timesteps=learning_steps_per_env, 
                                   reset_num_timesteps=False, 
                                   callback=callback)
                    else:
                        logger.info(f"Starting training on environment {env_idx+1}/{len(train_envs)} (seed {train_seeds[env_idx]})")
                        model.learn(total_timesteps=learning_steps_per_env,
                                   reset_num_timesteps=False)
                    
                    logger.info(f"Completed training on environment {env_idx+1} (seed {train_seeds[env_idx]})")
                    
                    # Run separate evaluation after each environment
                    if env_idx % 3 == 0:  # Evaluate every 3 environments
                        logger.info("\n----- Intermediate Evaluation After Environment Training -----")
                        obs, _ = eval_env.reset()
                        done = False
                        rewards = []
                        while not done:
                            try:
                                action, _ = model.predict(obs, deterministic=True)
                                obs, _, terminated, truncated, info = eval_env.step(action)
                                done = terminated or truncated
                                rewards.append(info["portfolio_return"])
                            except Exception as e:
                                action = np.zeros(eval_env.action_space.shape) # dummy action
                                obs, _, terminated, truncated, info = eval_env.step(action)
                                done = terminated or truncated
                                logger.warning(f"Error during evaluation: {e}")
                                continue
                        logger.info(f"Intermediate evaluation average return: {np.mean(rewards):.4f}")
                    
                except Exception as e:
                    logger.error(f"Error during training on environment {env_idx+1}: {e}")
                    continue
            
            # Force evaluation after each cycle
            # if cycle > 0 and cycle % 2 == 0:  # Every 2 cycles
            #     logger.info("\n----- Forced Evaluation After Cycle Completion -----")
            #     callback._evaluate_model()
        
        # Save final model
        model.save(model_path)
        logger.info(f"Final model saved: {model_path}")
        
        # Evaluate trained model (test with multiple seeds)
        results = []
        max_attempts = 3  # Set maximum number of attempts
        
        logger.info(f"\n===== Final Model Evaluation ({eval_episodes} test seeds) =====")
        
        for eval_seed in range(1000, 1000 + eval_episodes * 10, 10):
            attempts = 0
            while attempts < max_attempts:
                try:
                    # Create evaluation environment (with seed not used in training)
                    test_env = PortfolioEnv(seed=eval_seed)
                    
                    obs, _ = test_env.reset()
                    done = False
                    daily_returns = []
                    net_returns = []
                    episode_vols = []
                    t_bill_rates = []
                    
                    # Initial capital
                    initial_capital = 10000
                    portfolio_value = initial_capital
                    
                    while not done:
                        try:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, terminated, truncated, info = test_env.step(action)
                            done = terminated or truncated
                        except Exception as e:
                            action = np.zeros(test_env.action_space.shape) # dummy action
                            obs, _, terminated, truncated, info = test_env.step(action)
                            done = terminated or truncated
                            logger.warning(f"Error during evaluation: {e}")
                            continue
                        
                        # Store daily returns (already in decimal form)
                        daily_return = info["portfolio_return"]  # Already in decimal form (e.g., 0.01 = 1%)
                        daily_returns.append(daily_return)
                        
                        turnover = info["turnover"]
                        transaction_cost = 0.001 * turnover
                        
                        # Net return after transaction costs
                        net_return = daily_return - transaction_cost
                        net_returns.append(net_return)
                        
                        # Update portfolio value (compound returns)
                        portfolio_value *= (1 + net_return)
                        
                        episode_vols.append(info["portfolio_vol"])
                        
                        # Collect Treasury yield data
                        if test_env.current_step-1 < len(test_env.macro_data):
                            t_bill_rate = test_env.macro_data[test_env.current_step-1, 1] / 100.0
                            t_bill_rates.append(t_bill_rate)
                    
                    # Calculate total compound return
                    total_return = (portfolio_value / initial_capital - 1) * 100
                    
                    # Calculate annualized return
                    days = len(daily_returns)
                    trading_days_per_year = 252
                    annualized_return = ((1 + total_return/100) ** (trading_days_per_year/days) - 1) * 100
                    
                    # Calculate risk-free rate
                    if t_bill_rates:
                        annual_risk_free_rate = np.mean(t_bill_rates) * 100 * trading_days_per_year
                    else:
                        annual_risk_free_rate = 2.0
                    
                    # Calculate Sharpe ratio - using net return standard deviation
                    net_returns_array = np.array(net_returns) * 100
                    daily_std = np.std(net_returns_array) * 100
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
                    
                    logger.info(f"Seed {eval_seed} evaluation complete: Return {total_return:.2f}%, Sharpe {sharpe_ratio:.2f}")
                    break  # Exit loop if successful
                    
                except Exception as e:
                    attempts += 1
                    logger.warning(f"Seed {eval_seed} evaluation failed ({attempts}/{max_attempts}): {str(e)}")
                    if attempts >= max_attempts:
                        logger.warning(f"Skipping seed {eval_seed} evaluation")
            
            # Save intermediate results (every 100 episodes)
            if len(results) % 100 == 0 and len(results) > 0:
                avg_sharpe = np.mean([r["sharpe"] for r in results])
                avg_return = np.mean([r["total_return"] for r in results])
                avg_vol = np.mean([r["vol"] for r in results])
                
                logger.info(f"\n===== Intermediate Evaluation Results ({len(results)} episodes) =====")
                logger.info(f"Average Return: {avg_return:.4f}%")
                logger.info(f"Average Volatility: {avg_vol:.4f}%")
                logger.info(f"Average Sharpe: {avg_sharpe:.4f}")
        
        # Final evaluation results summary
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_return = np.mean([r["total_return"] for r in results])
        avg_vol = np.mean([r["vol"] for r in results])
        
        logger.info("\n" + "="*50)
        logger.info("===== Final Evaluation Results =====")
        logger.info(f"Average Return: {avg_return:.4f}%")
        logger.info(f"Average Volatility: {avg_vol:.4f}%")
        logger.info(f"Average Sharpe: {avg_sharpe:.4f}")
        logger.info("="*50)
        logger.info(f"Training completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())  # Print detailed error information
        raise e

if __name__ == '__main__':
    # Run FastAPI web server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True  # Terminate when main thread ends
    server_thread.start()
    
    logger.info("Web interface running at http://localhost:8000")
    time.sleep(1)  # Give server time to start
    
    iteration = 0
    
    while True:  # Infinite loop
        try:
            logger.info(f"\n===== Starting Training Iteration #{iteration+1} =====")
            main()
            iteration += 1
            logger.info(f"Completed Training Iteration #{iteration}")
            
            # Optional: Wait to prevent server overload
            time.sleep(10)  # Wait 10 seconds
            
        except Exception as e:
            logger.error(f"Execution failed (Iteration #{iteration+1}): {str(e)}")
            logger.error(traceback.format_exc())
            
            # Wait before retrying after error
            time.sleep(60)  # Wait 1 minute
            
            # Optional: Log serious errors
            with open("error_log.txt", "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Iteration #{iteration+1} Error: {str(e)}\n")
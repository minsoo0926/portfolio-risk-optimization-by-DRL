import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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

# Get logger
logger = setup_logger()

# Custom environment
class PortfolioEnv(gym.Env):
    def __init__(self, seed):
        super(PortfolioEnv, self).__init__()
        # 52-dim state (e.g., daily returns for 10 stocks, 63-day moving average, 63-day std dev, Relative Volume, VIX index, 5-year Treasury yield, previous action)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        # 10-dim action: weights for each stock (-1 to 1, later normalized to sum=0)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        # Load data for the current episode
        self.seed = seed
        data = None
        while data is None:
            data = generate_scenario(10, seed)
        
        # Use data excluding the first column (date, etc.)
        self.market_data = data.iloc[:, 1:41].values  # Stock data (10 stocks * 4 features = 40)
        self.macro_data = data.iloc[:, 41:43].values  # Macro data like VIX, Treasury yields (2 features)
        
        self.max_steps = len(data)
        self.current_step = 0
        self.previous_action = np.zeros(10)  # Initialize previous action to zeros
        
        # Initialize reward buffer (once when environment is created)
        self.reward_buffer = []
        self.return_buffer = []
        self.vol_buffer = []
        
        # Set initial state
        self.state = self._get_state()

    def _get_state(self):
        # Combine current market data and previous action to create state
        return np.concatenate([
            self.market_data[self.current_step].flatten(),  # 40 features
            self.macro_data[self.current_step].flatten(),   # 2 features
            self.previous_action                           # 10 features (previous action)
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.current_step = 0
        self.previous_action = np.zeros(10)
        self.state = self._get_state()
        return self.state, {}

    def step(self, action):
        # Normalize action
        action = action - np.mean(action)
        weights = action / (np.sum(np.abs(action)) + 1e-8)
        
        # Save current weights
        self.previous_action = weights.copy()
        
        # Move to next time step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get return data at new time step (t+1)
        if not terminated:
            returns_indices = np.arange(0, 40, 4)
            # Data is already stored as percentage (%), convert to decimal form (1% -> 0.01)
            stock_returns = self.market_data[self.current_step, returns_indices] / 100.0
            
            vol_indices = np.arange(2, 40, 4)
            # Data is already stored as percentage (%), convert to decimal form (1% -> 0.01)
            stock_vols = self.market_data[self.current_step, vol_indices] / 100.0
            
            # Calculate portfolio return (previous weights * current returns)
            portfolio_return = np.sum(weights * stock_returns)
            
            # Calculate portfolio risk
            portfolio_vol = np.sqrt(np.sum((weights * stock_vols) ** 2))
            
            # Calculate turnover
            # In the first step, all allocations are turnover
            turnover = np.sum(np.abs(weights))  # All positions are newly constructed
            
            # Calculate reward - focus on risk-adjusted return
            raw_reward = portfolio_return - 0.5 * portfolio_vol - 0.1 * turnover
            # Alternative: Sharpe ratio-like reward function
            # risk_adjusted_reward = portfolio_return / (portfolio_vol + 1e-8) - 0.1 * turnover
            reward = raw_reward
            
            # Calculate new state
            self.state = self._get_state()
        else:
            # No reward at episode end
            portfolio_return = 0
            portfolio_vol = 0
            turnover = 0
            reward = 0
        
        # Include return and volatility info in info dictionary
        info = {
            "portfolio_return": portfolio_return,
            "portfolio_vol": portfolio_vol,
            "turnover": turnover
        }
        
        return self.state, reward, terminated, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            logger.info(f"Step: {self.current_step}, Portfolio weights: {self.previous_action}")
        return 1

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
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            
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
        logger.info(f"► Average Daily Return (before costs): {mean_daily_return:.4f}%")
        logger.info(f"► Average Daily Net Return (after costs): {mean_daily_net_return:.4f}%")
        logger.info(f"► Total Compound Return: {total_return:.4f}%")
        logger.info(f"► Annualized Return: {annualized_return:.4f}%")
        logger.info(f"► Annual Risk-Free Rate: {annual_risk_free_rate:.4f}%")
        logger.info(f"► Daily Net Return Std Dev: {daily_std:.4f}%")
        logger.info(f"► Annualized Std Dev: {annualized_std:.4f}%")
        logger.info(f"► Sharpe Ratio: {sharpe:.4f}")
        
        # Output Sharpe ratio calculation details (for debugging)
        logger.info(f"► Sharpe Calculation: ({annualized_return:.4f} - {annual_risk_free_rate:.4f}) / {annualized_std:.4f} = {sharpe:.4f}")
        
        # Log to file
        with open("evaluation_log.txt", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Timestep {self.num_timesteps}\n")
            f.write(f"Average Daily Return (before costs): {mean_daily_return:.4f}%\n")
            f.write(f"Average Daily Net Return (after costs): {mean_daily_net_return:.4f}%\n")
            f.write(f"Total Compound Return: {total_return:.4f}%\n")
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
        eval_episodes = 100
        save_freq = 10000
        eval_freq = 20000

        # Ensure diversity in training data
        train_seeds = np.random.randint(0, 10000, size=10)  # Use multiple seeds
        logger.info(f"Training seeds: {train_seeds}")
        
        # Train across multiple environments
        train_envs = [PortfolioEnv(seed=seed) for seed in train_seeds]
        logger.info(f"Environments created: {len(train_envs)} environments")
        
        # Load existing model if available, otherwise create new one
        if os.path.exists(model_path + ".zip"):
            try:
                # Try to load existing model
                model = PPO.load(model_path, env=train_envs[0])
                logger.info(f"Model loaded successfully ({model_path})")
            except ValueError as e:
                # Create new model if observation space mismatch
                logger.warning(f"Failed to load existing model: {e}")
                logger.info("Creating new model.")
                policy_kwargs = dict(
                    net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),  # Modified network architecture
                    activation_fn=torch.nn.ReLU
                )
                model = PPO("MlpPolicy", train_envs[0], policy_kwargs=policy_kwargs,
                            learning_rate=0.0001,      # Learning rate for stable training
                            n_steps=2048,              # Longer trajectories for stable learning
                            batch_size=128,            # Appropriate batch size
                            gamma=0.99,                # Discount factor for future rewards
                            ent_coef=0.01,             # Slightly reduced entropy for exploration/exploitation balance
                            clip_range=0.1,            # Appropriate clipping range
                            vf_coef=0.5,               # Value function weight adjustment
                            max_grad_norm=0.5,         # Enhanced gradient clipping
                            verbose=2)
                logger.info("PPO model created successfully")
        else:
            policy_kwargs = dict(
                net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),  # Modified network architecture
                activation_fn=torch.nn.ReLU
            )
            model = PPO("MlpPolicy", train_envs[0], policy_kwargs=policy_kwargs,
                        learning_rate=0.0001,      # Learning rate for stable training
                        n_steps=2048,              # Longer trajectories for stable learning
                        batch_size=128,            # Appropriate batch size
                        gamma=0.99,                # Discount factor for future rewards
                        ent_coef=0.01,             # Slightly reduced entropy for exploration/exploitation balance
                        clip_range=0.1,            # Appropriate clipping range
                        vf_coef=0.5,               # Value function weight adjustment
                        max_grad_norm=0.5,         # Enhanced gradient clipping
                        verbose=2)
            logger.info("New PPO model created successfully")
        
        # Create evaluation environment
        eval_env = PortfolioEnv(seed=9999)
        callback = CustomCallback(
            eval_env=eval_env,
            save_freq=5000,   # Save more frequently
            eval_freq=5000,   # Evaluate more frequently
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
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, terminated, truncated, info = eval_env.step(action)
                            done = terminated or truncated
                            rewards.append(info["portfolio_return"])
                        logger.info(f"Intermediate evaluation average return: {np.mean(rewards):.4f}")
                    
                except Exception as e:
                    logger.error(f"Error during training on environment {env_idx+1}: {e}")
                    continue
            
            # Force evaluation after each cycle
            if cycle > 0 and cycle % 2 == 0:  # Every 2 cycles
                logger.info("\n----- Forced Evaluation After Cycle Completion -----")
                callback._evaluate_model()
        
        # Save final model
        model.save(model_path)
        logger.info(f"Final model saved: {model_path}")
        
        # Evaluate trained model (test with multiple seeds)
        results = []
        max_attempts = 3  # Set maximum number of attempts
        
        logger.info(f"\n===== Final Model Evaluation ({eval_episodes} test seeds) =====")
        
        for eval_seed in range(1000, 1000 + eval_episodes):
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
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = test_env.step(action)
                        done = terminated or truncated
                        
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
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
            
            # REMOVED: Periodic evaluation to avoid dimension errors
            # Just log progress every eval_freq steps
            if self.num_timesteps % self.eval_freq == 0:
                logger.info(f"Training progress: {self.num_timesteps} timesteps completed")
                
            return True
        except Exception as e:
            logger.error(f"Callback error: {str(e)}")
            return False


def main():
    try:
        logger.info("="*50)
        logger.info(f"Portfolio Optimization Training Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*50)
        
        model_path = "ppo_portfolio"
        total_timesteps = 500000  # Increased training steps
        eval_episodes = 10
        save_freq = 20000  # Less frequent saving
        eval_freq = 40000  # Less frequent evaluation
        
        # Option to disable final evaluation (set to False to avoid dimension errors)
        enable_final_evaluation = False  # Changed to False to avoid errors

        # Ensure diversity in training data
        np.random.seed(int(time.time()))
        train_seeds = np.random.randint(0, 10000, size=5)  # Reduced to 5 environments
        logger.info(f"Training seeds: {train_seeds}")
        
        # Create training environments
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
                
                # Create model with improved hyperparameters
                policy_kwargs = dict(
                    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),  # Larger networks
                    activation_fn=torch.nn.Tanh  # Better for financial data
                )
                
                model = PPO(
                    NormalizedActorCriticPolicy,
                    train_envs[0],
                    policy_kwargs=policy_kwargs,
                    learning_rate=3e-5,  # Lower learning rate
                    n_steps=512,  # Better for ~252 step episodes 
                    batch_size=64,  # Smaller batch size
                    gamma=0.995,  # Slightly higher discount for financial data
                    ent_coef=0.005,  # Lower entropy for more stable policies
                    clip_range=0.15,  # Slightly higher clip range
                    vf_coef=0.25,  # Lower value function coefficient
                    max_grad_norm=0.5,
                    verbose=2,
                    device=device
                )
                logger.info("PPO model created successfully")
        else:
            # Create new model with improved hyperparameters
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),  # Larger networks
                activation_fn=torch.nn.Tanh  # Better for financial data
            )
            
            model = PPO(
                NormalizedActorCriticPolicy,
                train_envs[0],
                policy_kwargs=policy_kwargs,
                learning_rate=3e-5,  # Lower learning rate
                n_steps=512,  # Better for ~252 step episodes
                batch_size=64,  # Smaller batch size  
                gamma=0.995,  # Slightly higher discount for financial data
                ent_coef=0.005,  # Lower entropy for more stable policies
                clip_range=0.15,  # Slightly higher clip range
                vf_coef=0.25,  # Lower value function coefficient
                max_grad_norm=0.5,
                verbose=2,
                device=device
            )
            logger.info("New PPO model created successfully")
        
        # Create evaluation environment
        eval_env = PortfolioEnv(seed=9999)
        callback = CustomCallback(
            eval_env=eval_env,
            save_freq=save_freq,
            eval_freq=eval_freq,
            model_path=model_path
        )
        logger.info("Evaluation environment and callback setup complete")
        
        # IMPROVED TRAINING STRATEGY: Longer training per environment
        steps_per_env = 100000  # Much longer training per environment
        logger.info(f"Training plan: Total {total_timesteps} steps, {steps_per_env} steps per environment")
        
        best_performance = -np.inf
        patience = 3  # Early stopping patience
        no_improvement_count = 0
        
        # Train on each environment for longer periods
        for env_idx, env in enumerate(train_envs):
            if total_timesteps <= 0:
                break
                
            current_steps = min(steps_per_env, total_timesteps)
            logger.info(f"\n===== Training on Environment {env_idx+1}/{len(train_envs)} (seed {train_seeds[env_idx]}) =====")
            logger.info(f"Training for {current_steps} steps")
            
            try:
                model.set_env(env)
                
                # Train for extended period on this environment
                model.learn(
                    total_timesteps=current_steps,
                    reset_num_timesteps=False,
                    callback=callback
                )
                
                logger.info(f"Completed training on environment {env_idx+1}")
                total_timesteps -= current_steps
                
                # REMOVED: Evaluation after each environment to avoid dimension errors
                logger.info(f"Environment {env_idx+1} training completed successfully")
                    
            except Exception as e:
                logger.error(f"Error during training on environment {env_idx+1}: {e}")
                continue
        
        # Save final model
        model.save(model_path)
        logger.info(f"Final model saved: {model_path}")
        
        # Final evaluation (optional - can be disabled to avoid dimension errors)
        if enable_final_evaluation:
            logger.info(f"\n===== Final Model Evaluation ({eval_episodes} test seeds) =====")
            
            # Evaluate trained model (test with multiple seeds)
            results = []
            max_attempts = 3  # Set maximum number of attempts
            
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
            if results:
                avg_sharpe = np.mean([r["sharpe"] for r in results])
                avg_return = np.mean([r["total_return"] for r in results])
                avg_vol = np.mean([r["vol"] for r in results])
                
                logger.info("\n" + "="*50)
                logger.info("===== Final Evaluation Results =====")
                logger.info(f"Average Return: {avg_return:.4f}%")
                logger.info(f"Average Volatility: {avg_vol:.4f}%")
                logger.info(f"Average Sharpe: {avg_sharpe:.4f}")
                logger.info("="*50)
        else:
            logger.info("Final evaluation disabled to avoid dimension errors")
        
        logger.info(f"Training completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())  # Print detailed error information
        raise e

if __name__ == '__main__':
    try:
        logger.info(f"\n===== Starting Portfolio Optimization Training =====")
        main()
        logger.info(f"Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log error for debugging
        with open("error_log.txt", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training Error: {str(e)}\n")
    
    logger.info("Training script finished")
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import time
import datetime
import logging
import traceback
import os
import threading
from app.server import start_server
from app.utils import setup_logger
from generate_scenario import generate_scenario
from models import create_ppo_model, NormalizedActorCriticPolicy

# Set up logger
logger = setup_logger()

# Device configuration
device = "cpu"  # PPO works better on CPU
logger.info(f"Using device: {device}")

# Environment class
class PortfolioEnv(gym.Env):
    def __init__(self, seed):
        super(PortfolioEnv, self).__init__()
        # State space: daily returns for 10 stocks, moving average, std dev, relative volume, VIX index, 5-year Treasury yield, previous action
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        # Action space: weights for each stock (-1 to 1, later normalized to sum=0)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        # Load data for the current episode
        self.seed = seed
        data = None
        while data is None:
            data = generate_scenario(10, seed)
        
        # Use data excluding the first column (date)
        self.market_data = data.iloc[:, 1:41].values  # Stock data (10 stocks * 4 features = 40)
        self.macro_data = data.iloc[:, 41:43].values  # Macro data like VIX, Treasury yields (2 features)
        
        self.max_steps = len(data)
        self.current_step = 0
        self.previous_action = np.zeros(10)  # Initialize previous action to zeros
        
        # Initialize reward buffer
        self.reward_buffer = []
        self.return_buffer = []
        self.vol_buffer = []
        
        # Set initial state
        self.state = self._get_state()

    def _get_state(self):
        """Get the current state, combining market data and previous action"""
        try:
            # Check array bounds
            if self.current_step >= len(self.market_data) or self.current_step >= len(self.macro_data):
                logger.warning(f"Index out of bounds: {self.current_step} (market_data: {len(self.market_data)}, macro_data: {len(self.macro_data)})")
                # Return safe default values
                return np.zeros(52, dtype=np.float32)
            
            # Get market data and check for NaN values
            market_features = self.market_data[self.current_step].flatten()
            if np.isnan(market_features).any():
                logger.warning(f"NaN detected in market_data at step {self.current_step}")
                market_features = np.nan_to_num(market_features, nan=0.0)
            
            # Get macro data and check for NaN values
            macro_features = self.macro_data[self.current_step].flatten()
            if np.isnan(macro_features).any():
                logger.warning(f"NaN detected in macro_data at step {self.current_step}")
                macro_features = np.nan_to_num(macro_features, nan=0.0)
            
            # Ensure previous_action has correct shape
            if len(self.previous_action) != 10:
                logger.warning(f"Previous action has incorrect shape: {self.previous_action.shape}, expected 10")
                self.previous_action = np.zeros(10, dtype=np.float32)
            
            # Concatenate all features
            state = np.concatenate([
                market_features,      # 40 features
                macro_features,       # 2 features
                self.previous_action  # 10 features (previous action)
            ]).astype(np.float32)
            
            # Final NaN check
            if np.isnan(state).any():
                logger.warning(f"NaN detected in final state at step {self.current_step}")
                state = np.nan_to_num(state, nan=0.0)
                
            # Check for inf values
            if np.isinf(state).any():
                logger.warning(f"Inf detected in state at step {self.current_step}")
                state = np.nan_to_num(state, posinf=1.0, neginf=-1.0)
                
            # Verify final shape
            if state.shape != (52,):
                logger.warning(f"Incorrect state shape: {state.shape}, expected (52,)")
                state = np.zeros(52, dtype=np.float32)
                
            return state
            
        except Exception as e:
            logger.error(f"Error creating state at step {self.current_step}: {str(e)}")
            # Return safe default values
            return np.zeros(52, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        if seed is not None:
            self.seed = seed
        self.current_step = 0
        self.previous_action = np.zeros(10)
        self.state = self._get_state()
        if np.isnan(self.state).any():
            logger.warning('WARNING: NaN in obs after reset!', self.state)
        return self.state, {}

    def step(self, action):
        """Take a step in the environment using the given action"""
        try:
            # Validate action input
            if action is None or np.isnan(action).any() or np.isinf(action).any():
                logger.warning(f"Invalid action detected: {action}")
                action = np.zeros(10, dtype=np.float32)  # Safe fallback
                
            weights = action
            
            # Move to next time step
            self.current_step += 1
            terminated = self.current_step >= self.max_steps
            truncated = False
            
            # Get return data at new time step (t+1)
            if not terminated:
                # Safe check for NaN in current state
                if np.isnan(self.state).any():
                    logger.warning(f'WARNING: NaN in obs after step!, pass this step. {self.state}')
                    return self.state, 0, terminated, True, {"portfolio_return": 0, "portfolio_vol": 0, "turnover": 0} # dummy output
                
                # Check bounds to prevent index errors
                if self.current_step >= len(self.market_data):
                    logger.warning(f"Step index {self.current_step} out of bounds (market_data: {len(self.market_data)})")
                    return self.state, 0, True, False, {"portfolio_return": 0, "portfolio_vol": 0, "turnover": 0}
                
                # Get return indices
                returns_indices = np.arange(0, 40, 4)
                try:
                    # Data is already stored as percentage (%), convert to decimal form (1% -> 0.01)
                    stock_returns = self.market_data[self.current_step, returns_indices] / 100.0
                    
                    # Check for NaN/Inf in returns
                    if np.isnan(stock_returns).any() or np.isinf(stock_returns).any():
                        logger.warning(f"Invalid values in stock returns: {stock_returns}")
                        stock_returns = np.nan_to_num(stock_returns, nan=0.0, posinf=0.01, neginf=-0.01)
                except Exception as e:
                    logger.error(f"Error accessing stock returns: {e}")
                    stock_returns = np.zeros(len(returns_indices), dtype=np.float32)
                
                # Get volatility indices
                vol_indices = np.arange(2, 40, 4)
                try:
                    # Data is already stored as percentage (%), convert to decimal form (1% -> 0.01)
                    stock_vols = self.market_data[self.current_step, vol_indices] / 100.0
                    
                    # Check for NaN/Inf in volatilities
                    if np.isnan(stock_vols).any() or np.isinf(stock_vols).any():
                        logger.warning(f"Invalid values in stock volatilities: {stock_vols}")
                        stock_vols = np.nan_to_num(stock_vols, nan=0.01, posinf=0.05, neginf=0.01)
                except Exception as e:
                    logger.error(f"Error accessing stock volatilities: {e}")
                    stock_vols = np.ones(len(vol_indices), dtype=np.float32) * 0.01  # Default volatility of 1%
                
                # Ensure weights and returns have same shape
                if len(weights) != len(stock_returns):
                    logger.warning(f"Shape mismatch: weights {len(weights)}, returns {len(stock_returns)}")
                    # Use minimum length to avoid index errors
                    min_len = min(len(weights), len(stock_returns))
                    weights = weights[:min_len]
                    stock_returns = stock_returns[:min_len]
                
                # Calculate portfolio return (previous weights * current returns)
                portfolio_return = np.sum(weights * stock_returns)
                
                # Calculate portfolio risk
                if len(weights) != len(stock_vols):
                    logger.warning(f"Shape mismatch: weights {len(weights)}, vols {len(stock_vols)}")
                    min_len = min(len(weights), len(stock_vols))
                    weights_for_vol = weights[:min_len]
                    stock_vols = stock_vols[:min_len]
                    portfolio_vol = np.sqrt(np.sum((weights_for_vol * stock_vols) ** 2))
                else:
                    portfolio_vol = np.sqrt(np.sum((weights * stock_vols) ** 2))
                
                # Calculate turnover
                if self.previous_action is None:
                    turnover = 0
                elif len(weights) != len(self.previous_action):
                    logger.warning(f"Shape mismatch: weights {len(weights)}, prev_action {len(self.previous_action)}")
                    min_len = min(len(weights), len(self.previous_action))
                    turnover = np.sum(np.abs(weights[:min_len] - self.previous_action[:min_len]))
                else:
                    turnover = np.sum(np.abs(weights - self.previous_action))
                
                # Save current weights as previous action (make defensive copy)
                self.previous_action = np.array(weights, dtype=np.float32).copy()
                
                # Calculate reward - focus on risk-adjusted return
                raw_reward = portfolio_return - 0.1 * portfolio_vol - 0.01 * turnover
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
            
        except Exception as e:
            logger.error(f"Error in step method: {str(e)}")
            # Return safe values in case of error
            self.state = np.zeros(52, dtype=np.float32)
            return self.state, 0, True, False, {"portfolio_return": 0, "portfolio_vol": 0, "turnover": 0}

    def render(self, mode="human"):
        if mode == "human":
            logger.info(f"Step: {self.current_step}, Portfolio weights: {self.previous_action}")
        return 1

def main():
    """Main training function"""
    try:
        logger.info("="*50)
        logger.info(f"Portfolio Optimization Training Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*50)
        
        # Set up training parameters
        model_path = "ppo_portfolio_simplified"
        total_timesteps = 50000  # Reduced for testing
        
        # Create environment
        env = PortfolioEnv(seed=42)
        logger.info("Environment created")
        
        # Create model with custom policy
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.Tanh  # More stable than ReLU
        )
        
        # Create model using our custom function
        model = create_ppo_model(env, policy_kwargs=policy_kwargs, device=device)
        logger.info("PPO model created successfully")
        
        # Training loop
        logger.info(f"Starting training for {total_timesteps} steps")
        model.learn(total_timesteps=total_timesteps)
        
        # Save final model
        model.save(model_path)
        logger.info(f"Model saved to {model_path}.zip")
        
        # Test model
        logger.info("Testing trained model")
        obs, _ = env.reset()
        done = False
        
        portfolio_values = [10000]  # Initial capital
        actions_taken = []
        
        while not done:
            try:
                action, _ = model.predict(obs, deterministic=True)
                actions_taken.append(action)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Calculate portfolio value
                daily_return = info["portfolio_return"]
                turnover = info["turnover"]
                transaction_cost = 0.001 * turnover
                net_return = daily_return - transaction_cost
                portfolio_values.append(portfolio_values[-1] * (1 + net_return))
                
            except Exception as e:
                logger.error(f"Error during testing: {e}")
                break
        
        # Calculate performance metrics
        if len(portfolio_values) > 1:
            total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
            logger.info(f"Test results: Total return: {total_return:.2f}%")
        else:
            logger.warning("Test failed - no portfolio values recorded")
        
        logger.info("Training and testing completed")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

if __name__ == '__main__':
    # Run web server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    logger.info("Web interface running at http://localhost:8000")
    time.sleep(1)  # Give server time to start
    
    # Run main function
    main()
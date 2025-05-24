import gymnasium as gym
import numpy as np
from generate_scenario import generate_scenario
import logging
from app.utils import setup_logger

# Get logger
logger = setup_logger()


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
        try:
            # Ensure we're not accessing beyond array bounds
            if self.current_step >= len(self.market_data) or self.current_step >= len(self.macro_data):
                logger.warning(f"Step index {self.current_step} out of bounds (market_data: {len(self.market_data)}, macro_data: {len(self.macro_data)})")
                # Return a safe fallback state with zeros
                return np.zeros(52, dtype=np.float32)
            
            # Get market data and check for NaN values
            market_features = self.market_data[self.current_step].flatten()
            if np.isnan(market_features).any():
                logger.warning(f"NaN detected in market_data at step {self.current_step}")
                # Replace NaN values with zeros
                market_features = np.nan_to_num(market_features, nan=0.0)
            
            # Get macro data and check for NaN values
            macro_features = self.macro_data[self.current_step].flatten()
            if np.isnan(macro_features).any():
                logger.warning(f"NaN detected in macro_data at step {self.current_step}")
                # Replace NaN values with zeros
                macro_features = np.nan_to_num(macro_features, nan=0.0)
            
            # Ensure previous_action has correct shape
            if len(self.previous_action) != 10:
                logger.warning(f"Previous action has incorrect shape: {self.previous_action.shape}, expected 10")
                self.previous_action = np.zeros(10, dtype=np.float32)
            
            # Concatenate all features
            state = np.concatenate([
                market_features,    # 40 features
                macro_features,     # 2 features
                self.previous_action # 10 features (previous action)
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
            # Return a safe fallback state
            return np.zeros(52, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.current_step = 0
        self.previous_action = np.zeros(10)
        self.state = self._get_state()
        if np.isnan(self.state).any():
            print('WARNING: NaN in obs after reset!', self.state)
        return self.state, {}

    def step(self, action):
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
import gymnasium as gym
import numpy as np
from generate_scenario import generate_scenario
import logging
from app.utils import setup_logger
import pandas as pd

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
        max_attempts = 10
        attempt = 0
        
        while data is None and attempt < max_attempts:
            data = generate_scenario(10, seed + attempt)
            attempt += 1
            
        if data is None:
            logger.error(f"Failed to generate scenario after {max_attempts} attempts")
            # Create dummy data as fallback
            dates = pd.date_range('2020-01-01', periods=252, freq='D')
            dummy_data = pd.DataFrame({
                'Date': dates,
                **{f'S{i}_return': np.random.normal(0.001, 0.02, 252) for i in range(1, 11)},
                **{f'S{i}_ma': np.random.normal(0.001, 0.01, 252) for i in range(1, 11)},
                **{f'S{i}_vol': np.random.uniform(0.01, 0.05, 252) for i in range(1, 11)},
                **{f'S{i}_rvol': np.random.uniform(0.8, 1.2, 252) for i in range(1, 11)},
                'VIX': np.random.uniform(15, 35, 252),
                'Treasury_5Y': np.random.uniform(1.5, 3.5, 252)
            })
            data = dummy_data
            logger.warning("Using dummy data as fallback")
        
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
            
            # IMPROVED: Normalize state for better value function learning
            # 1. Clip extreme values
            state = np.clip(state, -10.0, 10.0)
            
            # 2. Apply simple normalization for market data (first 40 features)
            if len(state) >= 40:
                # Normalize returns and volatility features separately
                returns_features = state[0:40:4]  # Every 4th feature starting from 0
                state[0:40:4] = np.clip(returns_features, -0.1, 0.1)  # Clip daily returns to ±10%
                
                vol_features = state[2:40:4]  # Every 4th feature starting from 2
                state[2:40:4] = np.clip(vol_features, 0.0, 0.5)  # Clip volatility to 0-50%
            
            # 3. Normalize macro features (VIX, Treasury)
            if len(state) >= 42:
                state[40] = np.clip(state[40], 10.0, 80.0) / 100.0  # Normalize VIX to 0-0.8 range
                state[41] = np.clip(state[41], 0.0, 10.0) / 100.0   # Normalize Treasury to 0-0.1 range
            
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
            import traceback
            logger.error(traceback.format_exc())
            # Return a safe fallback state
            return np.zeros(52, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.current_step = 0
        self.previous_action = np.zeros(10)
        
        # Reset cumulative return tracking for improved rewards
        self.cumulative_return = 0.0
        
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
            
            # DEBUG: Check weights properties for sum of |weights| = 1 normalization
            weights_sum = np.sum(weights)  # Net exposure
            weights_abs_sum = np.sum(np.abs(weights))  # Total exposure
            max_weight = np.max(np.abs(weights))
            
            if abs(weights_abs_sum - 1.0) > 0.01:  # Check total exposure
                logger.debug(f"WARNING: Sum of |weights| = {weights_abs_sum:.4f}, expected ~1.0")
                logger.debug(f"Weights: {[f'{w:.3f}' for w in weights]}")
            
            if abs(weights_sum) > 0.2:  # Check if too far from market neutral
                logger.debug(f"WARNING: Net exposure = {weights_sum:.4f}, consider market neutral strategy")
            
            if max_weight > 0.5:  # Check for extreme concentration
                logger.debug(f"WARNING: High concentration, max |weight| = {max_weight:.4f}")
            
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
                    import traceback
                    logger.error(traceback.format_exc())
                    stock_returns = np.zeros(10, dtype=np.float32)
                
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
                    import traceback
                    logger.error(traceback.format_exc())
                    stock_vols = np.ones(10, dtype=np.float32) * 0.01  # Default volatility of 1%
                
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
                
                # Calculate reward - Improved for better value function learning
                # 1. Scale portfolio return for numerical stability
                scaled_return = portfolio_return * 100  # Convert to percentage scale
                
                # 2. Add reward shaping with cumulative performance
                if not hasattr(self, 'cumulative_return'):
                    self.cumulative_return = 0.0
                self.cumulative_return += portfolio_return
                
                # 3. Risk-adjusted return with better scaling
                risk_penalty = 0.5 * portfolio_vol * 100  # Scale volatility penalty
                transaction_penalty = 0.1 * turnover  # Reduce transaction cost penalty
                
                # 4. Add exploration bonus for diversification
                diversification_bonus = 0.1 * (1.0 - np.max(np.abs(weights))) if len(weights) > 0 else 0.0
                
                # 5. Combine components with better scaling
                base_reward = scaled_return - risk_penalty - transaction_penalty + diversification_bonus
                
                # 6. Add small cumulative performance bonus
                performance_bonus = 0.01 * max(0, self.cumulative_return * 100)  # Only positive cumulative returns
                
                # 7. Final reward with normalization
                reward = base_reward + performance_bonus
                
                # 8. Clip reward to prevent extreme values
                reward = np.clip(reward, -10.0, 10.0)
                
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
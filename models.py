import torch as th
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
import logging

# Set up logger
logger = logging.getLogger(__name__)

class CustomMlpExtractor(nn.Module):
    """
    Custom MLP Extractor for actor-critic network with numerical stability optimizations.
    """
    def __init__(self, feature_dim, net_arch, activation_fn=nn.ReLU):
        super().__init__()
        
        # Define architecture for actor and critic networks
        policy_layers = []
        last_layer_dim_pi = feature_dim
        
        for layer_size in net_arch["pi"]:
            policy_layers.append(nn.Linear(last_layer_dim_pi, layer_size))
            policy_layers.append(activation_fn())
            last_layer_dim_pi = layer_size
        
        value_layers = []
        last_layer_dim_vf = feature_dim
        
        for layer_size in net_arch["vf"]:
            value_layers.append(nn.Linear(last_layer_dim_vf, layer_size))
            value_layers.append(activation_fn())
            last_layer_dim_vf = layer_size
        
        # Create networks
        self.policy_net = nn.Sequential(*policy_layers)
        self.value_net = nn.Sequential(*value_layers)
        
        # Better initialization for stability (less conservative)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use standard Xavier initialization
                nn.init.xavier_normal_(module.weight, gain=1.0)  # Increased gain from 0.01 to 1.0
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Logging
        print(f"Created CustomMlpExtractor with feature_dim={feature_dim}")
        print(f"Policy network: {self.policy_net}")
        print(f"Value network: {self.value_net}")
    
    def forward_actor(self, features):
        # Enhanced NaN checking
        if th.isnan(features).any() or th.isinf(features).any():
            logger.warning("NaN or Inf detected in actor features")
            features = th.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        output = self.policy_net(features)
        
        # Check output for NaN/Inf and clip to reasonable range
        if th.isnan(output).any() or th.isinf(output).any():
            logger.warning("NaN or Inf detected in actor output")
            output = th.nan_to_num(output, nan=0.0, posinf=2.0, neginf=-2.0)
        
        # Clip output to prevent extreme values
        output = th.clamp(output, min=-5.0, max=5.0)
        
        return output
    
    def forward_critic(self, features):
        # Enhanced NaN checking
        if th.isnan(features).any() or th.isinf(features).any():
            logger.warning("NaN or Inf detected in critic features")
            features = th.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        output = self.value_net(features)
        
        # Check output for NaN/Inf and clip to reasonable range
        if th.isnan(output).any() or th.isinf(output).any():
            logger.warning("NaN or Inf detected in critic output")
            output = th.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Clip output to prevent extreme values
        output = th.clamp(output, min=-20.0, max=20.0)
        
        return output
    
    def forward(self, features):
        """Do not call directly - maintained for compatibility"""
        return self.forward_actor(features), self.forward_critic(features)


class NormalizedActorCriticPolicy(ActorCriticPolicy):
    """
    Normalized Actor Critic Policy with better numerical stability
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,  
        *args,
        **kwargs,
    ):
        # Default network architecture
        if net_arch is None:
            net_arch = dict(pi=[128, 128, 64], vf=[128, 128, 64])
        
        # Initialize parent class
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        
        # Replace default MLP extractor with custom version
        feature_dim = self.features_extractor.features_dim
        self.mlp_extractor = CustomMlpExtractor(feature_dim, net_arch, activation_fn)
        
        # CRITICAL: Reinitialize action_net and value_net to prevent NaN issues
        self._reinitialize_networks()
        
        print("Initialized NormalizedActorCriticPolicy")
    
    def _reinitialize_networks(self):
        """Reinitialize the action and value networks with safe parameters"""
        # Reinitialize action_net (creates distribution parameters)
        if hasattr(self, 'action_net'):
            for layer in self.action_net.modules():
                if isinstance(layer, nn.Linear):
                    # Conservative initialization to prevent exploding gradients
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Reinitialize value_net
        if hasattr(self, 'value_net'):
            for layer in self.value_net.modules():
                if isinstance(layer, nn.Linear):
                    # Conservative initialization
                    nn.init.xavier_uniform_(layer.weight, gain=0.1) 
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _get_action_dist_from_latent(self, latent_pi):
        """Override to add NaN safety at distribution creation"""
        # Check input for NaN/Inf
        if th.isnan(latent_pi).any() or th.isinf(latent_pi).any():
            logger.warning("NaN/Inf in latent_pi for distribution creation")
            latent_pi = th.nan_to_num(latent_pi, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get mean actions from the policy network
        mean_actions = self.action_net(latent_pi)
        
        # Critical NaN check on mean_actions
        if th.isnan(mean_actions).any() or th.isinf(mean_actions).any():
            logger.warning("NaN/Inf detected in mean_actions, using safe fallback")
            batch_size = latent_pi.shape[0]
            mean_actions = th.zeros((batch_size, self.action_space.shape[0]), device=latent_pi.device)
        
        # Clamp mean actions to reasonable range
        mean_actions = th.clamp(mean_actions, min=-5.0, max=5.0)
        
        # Create distribution with fixed log_std to prevent NaN in scale
        if self.use_sde:
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        else:
            # Use a safe, fixed log_std to prevent scale issues
            safe_log_std = th.full_like(mean_actions, -1.0)  # std = exp(-1) ≈ 0.37
            
            # Ensure log_std doesn't have NaN/Inf
            if hasattr(self, 'log_std'):
                original_log_std = self.log_std
                if th.isnan(original_log_std).any() or th.isinf(original_log_std).any():
                    logger.warning("NaN/Inf in log_std, using safe fallback")
                    self.log_std.data = safe_log_std[0]  # Use safe values
                else:
                    # Clamp to reasonable range
                    self.log_std.data = th.clamp(original_log_std, min=-3.0, max=1.0)
            
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Override predict method to ensure normalization is applied
        """
        # Convert observation to tensor if needed
        if isinstance(observation, np.ndarray):
            obs_tensor = th.tensor(observation, dtype=th.float32)
        else:
            obs_tensor = observation
        
        # Ensure batch dimension
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # Use our custom forward method
        with th.no_grad():
            actions, _, _ = self.forward(obs_tensor, deterministic=deterministic)
        
        # Convert back to numpy and remove batch dimension if needed
        actions_np = actions.detach().cpu().numpy()
        if actions_np.shape[0] == 1:
            actions_np = actions_np.squeeze(0)
        
        return actions_np, state
    
    def forward(self, obs, deterministic=False):
        # Enhanced observation handling with better error checking
        if th.isnan(obs).any() or th.isinf(obs).any():
            logger.warning("NaN or Inf detected in observations, applying nan_to_num")
            obs = th.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Extract features
        features = self.extract_features(obs)
        
        # Check features for NaN/Inf
        if th.isnan(features).any() or th.isinf(features).any():
            logger.warning("NaN or Inf detected in features, applying nan_to_num")
            features = th.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Actor/critic networks
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        # Check latent representations for NaN/Inf
        if th.isnan(latent_pi).any() or th.isinf(latent_pi).any():
            logger.warning("NaN or Inf detected in latent_pi, applying nan_to_num")
            latent_pi = th.nan_to_num(latent_pi, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if th.isnan(latent_vf).any() or th.isinf(latent_vf).any():
            logger.warning("NaN or Inf detected in latent_vf, applying nan_to_num")
            latent_vf = th.nan_to_num(latent_vf, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Calculate values
        values = self.value_net(latent_vf)
        
        # Check values for NaN/Inf
        if th.isnan(values).any() or th.isinf(values).any():
            logger.warning("NaN or Inf detected in values, applying nan_to_num")
            values = th.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Action distribution - with safer parameters
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Check distribution parameters
        if hasattr(distribution.distribution, 'loc'):
            loc = distribution.distribution.loc
            scale = distribution.distribution.scale
            
            if th.isnan(loc).any() or th.isinf(loc).any():
                logger.warning("NaN or Inf detected in distribution loc")
                # Create a safe fallback action
                batch_size = obs.shape[0]
                safe_actions = th.zeros((batch_size, self.action_space.shape[0]), device=obs.device)
                safe_log_prob = th.zeros((batch_size,), device=obs.device)
                return safe_actions, values, safe_log_prob
                
            if th.isnan(scale).any() or th.isinf(scale).any() or (scale <= 0).any():
                logger.warning("NaN, Inf, or non-positive values detected in distribution scale")
                batch_size = obs.shape[0]
                safe_actions = th.zeros((batch_size, self.action_space.shape[0]), device=obs.device)
                safe_log_prob = th.zeros((batch_size,), device=obs.device)
                return safe_actions, values, safe_log_prob
        
        # Sample actions
        try:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
        except Exception as e:
            logger.warning(f"Error sampling actions: {e}")
            batch_size = obs.shape[0]
            safe_actions = th.zeros((batch_size, self.action_space.shape[0]), device=obs.device)
            safe_log_prob = th.zeros((batch_size,), device=obs.device)
            return safe_actions, values, safe_log_prob
        
        # Check sampled actions for NaN/Inf
        if th.isnan(actions).any() or th.isinf(actions).any():
            logger.warning("NaN or Inf detected in sampled actions")
            batch_size = obs.shape[0]
            safe_actions = th.zeros((batch_size, self.action_space.shape[0]), device=obs.device)
            safe_log_prob = th.zeros((batch_size,), device=obs.device)
            return safe_actions, values, safe_log_prob
        
        # IMPROVED Portfolio weights normalization with better error handling
        try:
            # Step 1: Apply tanh to bound the actions first (more stable)
            bounded_actions = th.tanh(actions)
            
            # Step 2: Center the actions to have mean 0
            actions_mean = th.mean(bounded_actions, dim=1, keepdim=True)
            centered_actions = bounded_actions - actions_mean
            
            # Step 3: Normalize so sum of absolute values = 1 (total exposure = 1)
            abs_sum = th.sum(th.abs(centered_actions), dim=1, keepdim=True)
            abs_sum = th.clamp(abs_sum, min=1e-6)  # Increased minimum to prevent division issues
            
            normalized_actions = centered_actions / abs_sum
            
            # Final safety check
            if th.isnan(normalized_actions).any() or th.isinf(normalized_actions).any():
                logger.warning("NaN or Inf detected in normalized actions, using zero actions")
                batch_size = obs.shape[0]
                normalized_actions = th.zeros((batch_size, self.action_space.shape[0]), device=obs.device)
            
            # Reshape actions
            final_actions = normalized_actions.reshape((-1, *self.action_space.shape))
            
        except Exception as e:
            logger.warning(f"Error in action normalization: {e}")
            batch_size = obs.shape[0]
            final_actions = th.zeros((batch_size, self.action_space.shape[0]), device=obs.device)
        
        return final_actions, values, log_prob


def create_ppo_model(env, policy_kwargs=None, device="cpu"):
    """
    Function to create a PPO model optimized for portfolio optimization
    """
    from stable_baselines3 import PPO
    
    # Default policy keyword arguments optimized for portfolio weights
    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64, 32], vf=[64, 64, 32]),  # Larger networks
            activation_fn=nn.ReLU  # Better for portfolio optimization
        )
    
    # Create PPO model with parameters tuned for portfolio optimization
    model = PPO(
        NormalizedActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-5,  # Lower learning rate for stability
        n_steps=512,  # Suitable for ~252 step episodes
        batch_size=64,  # Smaller batch size for better gradient estimates
        gamma=0.995,  # Slightly higher discount for financial data
        ent_coef=0.01,  # Moderate entropy for exploration
        clip_range=0.15,  # Slightly higher clip range
        vf_coef=0.25,  # Lower value function coefficient
        max_grad_norm=0.5,  # Gradient clipping for stability
        verbose=2,
        device=device
    )
    
    print(f"Created PPO model with portfolio-optimized parameters on device: {device}")
    return model
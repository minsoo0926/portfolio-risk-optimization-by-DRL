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
        
        # Conservative initialization for stability
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization (suitable for Tanh activation)
                nn.init.xavier_normal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Logging
        print(f"Created CustomMlpExtractor with feature_dim={feature_dim}")
        print(f"Policy network: {self.policy_net}")
        print(f"Value network: {self.value_net}")
    
    def forward_actor(self, features):
        # Check for NaN values
        if th.isnan(features).any():
            features = th.nan_to_num(features)
        
        return self.policy_net(features)
    
    def forward_critic(self, features):
        # Check for NaN values
        if th.isnan(features).any():
            features = th.nan_to_num(features)
            
        return self.value_net(features)
    
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
        
        print("Initialized NormalizedActorCriticPolicy")
    
    def forward(self, obs, deterministic=False):
        # Safe observation handling
        if th.isnan(obs).any():
            obs = th.nan_to_num(obs)
        
        # Extract features
        features = self.extract_features(obs)
        
        # Actor/critic networks
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        # Calculate values
        values = self.value_net(latent_vf)
        
        # Action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Sample actions
        actions = distribution.get_actions(deterministic=deterministic)
        
        # IMPORTANT: Calculate log probabilities BEFORE any transformation
        log_prob = distribution.log_prob(actions)
        
        # Center actions
        actions_mean = th.mean(actions, dim=1, keepdim=True)
        centered_actions = actions - actions_mean
        
        # Normalize to make absolute values sum to 1
        abs_sum = th.sum(th.abs(centered_actions), dim=1, keepdim=True)
        abs_sum = th.clamp(abs_sum, min=1e-8)  # Prevent division by zero
        
        # Normalized actions
        normalized_actions = centered_actions / abs_sum
        
        # Reshape actions
        actions = normalized_actions.reshape((-1, *self.action_space.shape))
        
        return actions, values, log_prob


def create_ppo_model(env, policy_kwargs=None, device="cpu"):
    """
    Function to create a PPO model
    """
    from stable_baselines3 import PPO
    
    # Default policy keyword arguments
    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),
            activation_fn=nn.ReLU
        )
    
    # Create PPO model
    model = PPO(
        NormalizedActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=2,
        device=device
    )
    
    print(f"Created PPO model with custom policy on device: {device}")
    return model
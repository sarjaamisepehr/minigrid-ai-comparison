"""Neural network architectures for model-free agents."""
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    output_activation: str = "none"
) -> nn.Sequential:
    """
    Create a multi-layer perceptron.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ("relu", "tanh", "elu")
        output_activation: Output activation ("none", "softmax", "tanh")
        
    Returns:
        MLP as nn.Sequential
    """
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU
    }
    
    act_fn = activations.get(activation, nn.ReLU)
    
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_fn())
        prev_dim = hidden_dim
        
    layers.append(nn.Linear(prev_dim, output_dim))
    
    if output_activation == "softmax":
        layers.append(nn.Softmax(dim=-1))
    elif output_activation == "tanh":
        layers.append(nn.Tanh())
        
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """
    Policy network for discrete action spaces.
    
    Outputs action probabilities using softmax.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu"
    ):
        """
        Initialize policy network.
        
        Args:
            observation_dim: Dimension of flattened observation
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        self.network = create_mlp(
            input_dim=observation_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation="none"  # We'll apply softmax in forward
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
                
        # Smaller initialization for output layer
        last_layer = list(self.network.modules())[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.orthogonal_(last_layer.weight, gain=0.01)
            
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits.
        
        Args:
            observation: Observation tensor (batch_size, obs_dim)
            
        Returns:
            Action logits (batch_size, action_dim)
        """
        return self.network(observation)
    
    def get_distribution(self, observation: torch.Tensor) -> Categorical:
        """
        Get action distribution for observation.
        
        Args:
            observation: Observation tensor
            
        Returns:
            Categorical distribution over actions
        """
        logits = self.forward(observation)
        return Categorical(logits=logits)
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            observation: Observation tensor
            deterministic: If True, return argmax action
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Entropy of distribution
        """
        dist = self.get_distribution(observation)
        
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities of actions.
        
        Args:
            observations: Batch of observations
            actions: Batch of actions taken
            
        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of distributions
        """
        dist = self.get_distribution(observations)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """
    Value network (critic) estimating state values.
    """
    
    def __init__(
        self,
        observation_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu"
    ):
        """
        Initialize value network.
        
        Args:
            observation_dim: Dimension of flattened observation
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        
        self.network = create_mlp(
            input_dim=observation_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation="none"
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
                
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.
        
        Args:
            observation: Observation tensor (batch_size, obs_dim)
            
        Returns:
            State value (batch_size, 1)
        """
        return self.network(observation)


class ActorCriticNetwork(nn.Module):
    """
    Combined actor-critic network with shared feature extractor.
    
    More parameter efficient than separate networks.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
        shared_layers: int = 1
    ):
        """
        Initialize combined actor-critic network.
        
        Args:
            observation_dim: Dimension of flattened observation
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            shared_layers: Number of shared hidden layers
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU
        }
        act_fn = activations.get(activation, nn.ReLU)
        
        # Shared feature extractor
        shared_dims = hidden_dims[:shared_layers]
        self.shared = self._build_layers(
            observation_dim, 
            shared_dims, 
            act_fn
        )
        
        # Compute shared output dimension
        shared_out_dim = shared_dims[-1] if shared_dims else observation_dim
        
        # Policy head
        policy_dims = hidden_dims[shared_layers:]
        if policy_dims:
            self.policy_head = nn.Sequential(
                self._build_layers(shared_out_dim, policy_dims, act_fn),
                nn.Linear(policy_dims[-1], action_dim)
            )
        else:
            self.policy_head = nn.Linear(shared_out_dim, action_dim)
            
        # Value head
        value_dims = hidden_dims[shared_layers:]
        if value_dims:
            self.value_head = nn.Sequential(
                self._build_layers(shared_out_dim, value_dims, act_fn),
                nn.Linear(value_dims[-1], 1)
            )
        else:
            self.value_head = nn.Linear(shared_out_dim, 1)
            
        self._init_weights()
        
    def _build_layers(
        self, 
        input_dim: int, 
        hidden_dims: List[int],
        act_fn
    ) -> nn.Sequential:
        """Build sequential layers."""
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(act_fn())
            prev_dim = dim
            
        return nn.Sequential(*layers) if layers else nn.Identity()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
                
    def forward(
        self, 
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy logits and value.
        
        Args:
            observation: Observation tensor
            
        Returns:
            policy_logits: Action logits
            value: State value
        """
        features = self.shared(observation)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
    
    def get_action_and_value(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log prob, entropy, and value in single forward pass.
        
        Args:
            observation: Observation tensor
            deterministic: If True, return argmax action
            
        Returns:
            action, log_prob, entropy, value
        """
        policy_logits, value = self.forward(observation)
        dist = Categorical(logits=policy_logits)
        
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO-style updates.
        
        Args:
            observations: Batch of observations
            actions: Batch of actions
            
        Returns:
            log_probs, entropy, values
        """
        policy_logits, values = self.forward(observations)
        dist = Categorical(logits=policy_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, values.squeeze(-1)


# Need numpy for weight initialization
import numpy as np
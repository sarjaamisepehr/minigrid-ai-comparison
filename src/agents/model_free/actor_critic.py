"""Actor-Critic (A2C) agent implementation for discrete action spaces."""
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..base_agent import BaseAgent, AgentOutput
from .networks import ActorCriticNetwork


class ActorCriticAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C) agent for discrete action spaces.
    
    Uses n-step returns for value estimation and entropy regularization
    for exploration. This is a synchronous, model-free baseline.
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """
        Initialize A2C agent.
        
        Args:
            observation_shape: Shape of observations
            action_dim: Number of discrete actions
            config: Configuration with keys:
                - network.hidden_dims: List of hidden layer sizes
                - network.activation: Activation function
                - learning_rate: Optimizer learning rate
                - gamma: Discount factor
                - entropy_coef: Entropy bonus coefficient
                - value_coef: Value loss coefficient
                - max_grad_norm: Gradient clipping threshold
                - n_steps: Steps before update (for n-step returns)
            device: Torch device
        """
        super().__init__(observation_shape, action_dim, config, device)
        
        # Flatten observation dimension
        self.observation_dim = int(np.prod(observation_shape))
        
        # Extract config values
        network_config = config.get("network", {})
        self.hidden_dims = network_config.get("hidden_dims", [128, 128])
        self.activation = network_config.get("activation", "relu")
        
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.n_steps = config.get("n_steps", 5)
        
        # Build network
        self.network = ActorCriticNetwork(
            observation_dim=self.observation_dim,
            action_dim=action_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.lr
        )
        
        # Storage for n-step collection
        self._reset_storage()
        
    def _reset_storage(self):
        """Reset n-step storage buffers."""
        self.storage = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": []
        }
        
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> AgentOutput:
        """
        Select action given observation.
        
        Args:
            observation: Current observation
            deterministic: If True, select greedy action
            
        Returns:
            AgentOutput with action and auxiliary info
        """
        # Convert and flatten observation
        obs_tensor = self._to_tensor(observation).unsqueeze(0)
        if len(obs_tensor.shape) > 2:
            obs_tensor = obs_tensor.view(1, -1)
            
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(
                obs_tensor,
                deterministic=deterministic
            )
            
        return AgentOutput(
            action=action.item(),
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            info={"value": value.item()}
        )
    
    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: torch.Tensor
    ):
        """
        Store transition for n-step update.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            done: Whether episode ended
            value: Value estimate
            log_prob: Log probability of action
        """
        self.storage["observations"].append(observation)
        self.storage["actions"].append(action)
        self.storage["rewards"].append(reward)
        self.storage["dones"].append(done)
        self.storage["values"].append(value)
        self.storage["log_probs"].append(log_prob)
        
    def should_update(self) -> bool:
        """Check if enough steps collected for update."""
        return len(self.storage["observations"]) >= self.n_steps
    
    def compute_returns(
        self,
        next_value: float,
        rewards: list,
        dones: list,
        values: list
    ) -> torch.Tensor:
        """
        Compute n-step returns with GAE-like advantage estimation.
        
        Args:
            next_value: Value estimate for state after last stored state
            rewards: List of rewards
            dones: List of done flags
            values: List of value estimates
            
        Returns:
            Tensor of computed returns
        """
        returns = []
        R = next_value
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
            
        return torch.tensor(returns, dtype=torch.float32, device=self.device)
    
    def learn(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        next_observation: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Update agent from collected n-step experience.
        
        Args:
            batch: Optional pre-batched data (unused in n-step A2C)
            next_observation: Observation after last stored transition
            
        Returns:
            Dictionary of loss values
        """
        if not self.should_update():
            return {}
            
        # Get next value for bootstrapping
        if next_observation is not None:
            obs_tensor = self._to_tensor(next_observation).unsqueeze(0)
            if len(obs_tensor.shape) > 2:
                obs_tensor = obs_tensor.view(1, -1)
            with torch.no_grad():
                _, _, _, next_value = self.network.get_action_and_value(obs_tensor)
                next_value = next_value.item()
        else:
            next_value = 0.0
            
        # Compute returns
        returns = self.compute_returns(
            next_value,
            self.storage["rewards"],
            self.storage["dones"],
            self.storage["values"]
        )
        
        # Convert storage to tensors
        observations = torch.stack([
            self._to_tensor(obs).view(-1) 
            for obs in self.storage["observations"]
        ])
        actions = torch.tensor(
            self.storage["actions"], 
            dtype=torch.long, 
            device=self.device
        )
        old_log_probs = torch.stack(self.storage["log_probs"]).squeeze()
        old_values = torch.tensor(
            self.storage["values"],
            dtype=torch.float32,
            device=self.device
        )
        
        # Forward pass
        log_probs, entropy, values = self.network.evaluate_actions(
            observations, actions
        )
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Normalize advantages (optional but often helpful)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Entropy bonus (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss 
            + self.value_coef * value_loss 
            + self.entropy_coef * entropy_loss
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.max_grad_norm
            )
            
        self.optimizer.step()
        
        # Update statistics
        self.total_steps += len(self.storage["observations"])
        
        # Compute metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": total_loss.item(),
            "mean_value": values.mean().item(),
            "mean_return": returns.mean().item(),
            "mean_advantage": advantages.mean().item()
        }
        
        # Clear storage
        self._reset_storage()
        
        return metrics
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save agent state.
        
        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes
        }, path)
        
    def load(self, path: Union[str, Path]) -> None:
        """
        Load agent state.
        
        Args:
            path: Path to saved file
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.total_episodes = checkpoint.get("total_episodes", 0)
        
    def train_mode(self) -> None:
        """Set to training mode."""
        super().train_mode()
        self.network.train()
        
    def eval_mode(self) -> None:
        """Set to evaluation mode."""
        super().eval_mode()
        self.network.eval()
"""Abstract base class for all agents."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch


@dataclass
class AgentOutput:
    """Standardized output from agent's act method."""
    action: Union[int, np.ndarray]
    log_prob: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    info: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """
    Abstract base class defining interface for all agents.
    
    Ensures consistent API between Active Inference and RL agents
    for fair comparison in experiments.
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """
        Initialize agent.
        
        Args:
            observation_shape: Shape of observations
            action_dim: Number of actions (discrete) or action dimensions (continuous)
            config: Agent configuration dictionary
            device: Torch device ("cpu" or "cuda")
        """
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device)
        
        # Track training statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.training = True
        
    @abstractmethod
    def act(
        self, 
        observation: np.ndarray,
        deterministic: bool = False
    ) -> AgentOutput:
        """
        Select action given observation.
        
        Args:
            observation: Current observation
            deterministic: If True, select most likely action (for evaluation)
            
        Returns:
            AgentOutput containing action and auxiliary information
        """
        pass
    
    @abstractmethod
    def learn(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Update agent from batch of experience.
        
        Args:
            batch: Dictionary containing:
                - observations: (batch_size, *obs_shape)
                - actions: (batch_size,) or (batch_size, action_dim)
                - rewards: (batch_size,)
                - next_observations: (batch_size, *obs_shape)
                - dones: (batch_size,)
                
        Returns:
            Dictionary of loss values and metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save agent state to file."""
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load agent state from file."""
        pass
    
    def train_mode(self) -> None:
        """Set agent to training mode."""
        self.training = True
        
    def eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        self.training = False
        
    def _to_tensor(
        self, 
        x: np.ndarray, 
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Convert numpy array to tensor on correct device."""
        return torch.tensor(x, dtype=dtype, device=self.device)
    
    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        return x.detach().cpu().numpy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics for logging."""
        return {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "training": self.training
        }
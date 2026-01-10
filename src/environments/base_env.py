"""Abstract base class for all environments."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


@dataclass
class EnvSpec:
    """Environment specification containing key properties."""
    observation_shape: Tuple[int, ...]
    action_space_type: str  # "discrete" or "continuous"
    action_dim: int
    action_low: Optional[np.ndarray] = None  # For continuous
    action_high: Optional[np.ndarray] = None  # For continuous
    max_steps: int = 100


class BaseEnvironment(ABC):
    """
    Abstract base class defining the interface for all environments.
    
    This ensures consistent API across discrete (MiniGrid) and 
    continuous (DMControl) environments for fair agent comparison.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize environment with configuration.
        
        Args:
            config: Environment configuration dictionary
        """
        self.config = config
        self._env = None
        self._spec: Optional[EnvSpec] = None
        
    @property
    def spec(self) -> EnvSpec:
        """Return environment specification."""
        if self._spec is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._spec
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        pass
    
    @abstractmethod
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take action in environment.
        
        Args:
            action: Action to take (int for discrete, array for continuous)
            
        Returns:
            observation: Next observation
            reward: Reward received
            terminated: Whether episode ended due to task completion/failure
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        pass
    
    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """Render current state."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """Get shape of observations."""
        return self.spec.observation_shape
    
    def get_action_dim(self) -> int:
        """Get dimension of action space."""
        return self.spec.action_dim
    
    def is_discrete(self) -> bool:
        """Check if action space is discrete."""
        return self.spec.action_space_type == "discrete"
    
    def sample_random_action(self) -> Union[int, np.ndarray]:
        """Sample random action from action space."""
        if self.is_discrete():
            return np.random.randint(0, self.spec.action_dim)
        else:
            return np.random.uniform(
                self.spec.action_low, 
                self.spec.action_high
            )
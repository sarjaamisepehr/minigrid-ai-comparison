"""MiniGrid environment wrapper with modular observation processing."""
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import (
    ImgObsWrapper, 
    FlatObsWrapper,
    FullyObsWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper
)

from .base_env import BaseEnvironment, EnvSpec


class MiniGridEnvironment(BaseEnvironment):
    """
    MiniGrid environment wrapper providing consistent interface.
    
    Supports various observation types and wrapper configurations
    for fair comparison between agents.
    
    MiniGrid Action Space (Discrete 7):
        0: Turn left
        1: Turn right
        2: Move forward
        3: Pick up object
        4: Drop object
        5: Toggle (open/close door)
        6: Done (not used in most envs)
    """
    
    # MiniGrid action mapping for reference
    ACTIONS = {
        0: "turn_left",
        1: "turn_right", 
        2: "forward",
        3: "pickup",
        4: "drop",
        5: "toggle",
        6: "done"
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MiniGrid environment.
        
        Args:
            config: Configuration with keys:
                - env_id: MiniGrid environment ID
                - observation_type: "image", "flat", "rgb", "rgb_partial"
                - fully_observable: Whether agent sees full grid
                - max_steps: Maximum steps per episode
                - render_mode: None, "human", or "rgb_array"
        """
        super().__init__(config)
        
        self.env_id = config.get("env_id", "MiniGrid-Empty-8x8-v0")
        self.observation_type = config.get("observation_type", "flat")
        self.fully_observable = config.get("fully_observable", False)
        self.max_steps = config.get("max_steps", 100)
        self.render_mode = config.get("render_mode", None)
        
        self._build_environment()
        
    def _build_environment(self) -> None:
        """Build environment with appropriate wrappers."""
        # Create base environment
        self._env = gym.make(
            self.env_id,
            max_steps=self.max_steps,
            render_mode=self.render_mode
        )
        
        # Apply full observability wrapper if requested
        if self.fully_observable:
            self._env = FullyObsWrapper(self._env)
        
        # Apply observation type wrapper
        if self.observation_type == "image":
            # Returns compact symbolic image (7x7x3 by default)
            self._env = ImgObsWrapper(self._env)
        elif self.observation_type == "flat":
            # Returns flattened vector observation
            self._env = FlatObsWrapper(self._env)
        elif self.observation_type == "rgb":
            # Returns RGB image of full grid
            self._env = RGBImgObsWrapper(self._env)
        elif self.observation_type == "rgb_partial":
            # Returns RGB image of agent's view
            self._env = RGBImgPartialObsWrapper(self._env)
        else:
            # Default: ImgObsWrapper for symbolic observation
            self._env = ImgObsWrapper(self._env)
            
        # Build environment specification
        self._build_spec()
        
    def _build_spec(self) -> None:
        """Build environment specification from observation/action spaces."""
        obs_space = self._env.observation_space
        act_space = self._env.action_space
        
        # Handle different observation space types
        if isinstance(obs_space, gym.spaces.Box):
            obs_shape = obs_space.shape
        elif isinstance(obs_space, gym.spaces.Dict):
            # For dict observations, use 'image' key shape
            obs_shape = obs_space['image'].shape
        else:
            obs_shape = (obs_space.n,)  # For discrete (shouldn't happen)
            
        # Flatten observation shape for consistency
        if len(obs_shape) > 1 and self.observation_type == "flat":
            obs_shape = (int(np.prod(obs_shape)),)
            
        self._spec = EnvSpec(
            observation_shape=obs_shape,
            action_space_type="discrete",
            action_dim=act_space.n,
            max_steps=self.max_steps
        )
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observation: Initial observation as numpy array
            info: Information dictionary with 'direction', 'mission', etc.
        """
        obs, info = self._env.reset(seed=seed)
        obs = self._process_observation(obs)
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take action in environment.
        
        Args:
            action: Discrete action (0-6)
            
        Returns:
            observation: Next observation
            reward: Reward (1 - 0.9 * step_count/max_steps on success, 0 otherwise)
            terminated: True if goal reached or failed
            truncated: True if max steps exceeded
            info: Additional information
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = self._process_observation(obs)
        return obs, reward, terminated, truncated, info
    
    def _process_observation(self, obs: Any) -> np.ndarray:
        """
        Process observation to consistent numpy array format.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            Processed observation as float32 numpy array
        """
        if isinstance(obs, dict):
            obs = obs['image']
            
        obs = np.asarray(obs, dtype=np.float32)
        
        # Normalize pixel values if RGB
        if self.observation_type in ["rgb", "rgb_partial"]:
            obs = obs / 255.0
            
        # Flatten if needed
        if self.observation_type == "flat" and len(obs.shape) > 1:
            obs = obs.flatten()
            
        return obs
    
    def render(self) -> Optional[np.ndarray]:
        """Render current state."""
        return self._env.render()
    
    def close(self) -> None:
        """Close environment."""
        if self._env is not None:
            self._env.close()
            
    def get_action_meanings(self) -> Dict[int, str]:
        """Get human-readable action meanings."""
        return self.ACTIONS.copy()
    
    def __repr__(self) -> str:
        return (
            f"MiniGridEnvironment("
            f"env_id={self.env_id}, "
            f"obs_type={self.observation_type}, "
            f"fully_obs={self.fully_observable})"
        )


class CustomMiniGridEnv(MiniGridEnvironment):
    """
    Factory for creating custom MiniGrid configurations.
    
    Provides preset configurations for common experimental setups.
    """
    
    PRESETS = {
        "empty_8x8": {
            "env_id": "MiniGrid-Empty-8x8-v0",
            "observation_type": "flat",
            "fully_observable": False,
            "max_steps": 100
        },
        "door_key_8x8": {
            "env_id": "MiniGrid-DoorKey-8x8-v0",
            "observation_type": "flat",
            "fully_observable": False,
            "max_steps": 150
        },
        "key_corridor_s3r1": {
            "env_id": "MiniGrid-KeyCorridorS3R1-v0",
            "observation_type": "flat",
            "fully_observable": False,
            "max_steps": 200
        },
        "four_rooms": {
            "env_id": "MiniGrid-FourRooms-v0",
            "observation_type": "flat",
            "fully_observable": False,
            "max_steps": 200
        },
    }
    
    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'CustomMiniGridEnv':
        """
        Create environment from preset configuration.
        
        Args:
            preset_name: Name of preset configuration
            **overrides: Override specific config values
            
        Returns:
            Configured MiniGrid environment
        """
        if preset_name not in cls.PRESETS:
            available = list(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
            
        config = cls.PRESETS[preset_name].copy()
        config.update(overrides)
        return cls(config)
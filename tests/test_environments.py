"""Tests for environment implementations."""
import pytest
import numpy as np

from src.environments import MiniGridEnvironment
from src.environments.minigrid_env import CustomMiniGridEnv


class TestMiniGridEnvironment:
    """Test suite for MiniGrid environment wrapper."""
    
    @pytest.fixture
    def default_config(self):
        return {
            "env_id": "MiniGrid-Empty-8x8-v0",
            "observation_type": "flat",
            "fully_observable": False,
            "max_steps": 100,
            "render_mode": None
        }
    
    def test_initialization(self, default_config):
        """Test environment initializes correctly."""
        env = MiniGridEnvironment(default_config)
        assert env is not None
        env.close()
        
    def test_reset(self, default_config):
        """Test reset returns valid observation and info."""
        env = MiniGridEnvironment(default_config)
        obs, info = env.reset(seed=42)
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        
        env.close()
        
    def test_step(self, default_config):
        """Test step returns correct tuple structure."""
        env = MiniGridEnvironment(default_config)
        env.reset(seed=42)
        
        # Take a forward action
        obs, reward, terminated, truncated, info = env.step(2)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()
        
    def test_observation_shape_flat(self, default_config):
        """Test flat observation has correct shape."""
        env = MiniGridEnvironment(default_config)
        obs, _ = env.reset()
        
        assert len(obs.shape) == 1  # Should be flattened
        assert obs.shape == env.spec.observation_shape
        
        env.close()
        
    def test_observation_shape_image(self, default_config):
        """Test image observation has correct shape."""
        default_config["observation_type"] = "image"
        env = MiniGridEnvironment(default_config)
        obs, _ = env.reset()
        
        assert len(obs.shape) == 3  # (height, width, channels)
        
        env.close()
        
    def test_discrete_action_space(self, default_config):
        """Test action space is discrete with correct dimension."""
        env = MiniGridEnvironment(default_config)
        env.reset()
        
        assert env.is_discrete()
        assert env.spec.action_dim == 7  # MiniGrid has 7 actions
        
        env.close()
        
    def test_random_action_sampling(self, default_config):
        """Test random action sampling returns valid actions."""
        env = MiniGridEnvironment(default_config)
        env.reset()
        
        for _ in range(10):
            action = env.sample_random_action()
            assert 0 <= action < env.spec.action_dim
            
        env.close()
        
    def test_episode_termination(self, default_config):
        """Test episode terminates correctly."""
        env = MiniGridEnvironment(default_config)
        env.reset(seed=42)
        
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = env.sample_random_action()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
        assert done  # Should terminate within max_steps
        
        env.close()
        

class TestCustomMiniGridEnv:
    """Test suite for preset configurations."""
    
    def test_from_preset(self):
        """Test creating environment from preset."""
        env = CustomMiniGridEnv.from_preset("empty_8x8")
        obs, _ = env.reset()
        
        assert obs is not None
        env.close()
        
    def test_preset_with_override(self):
        """Test preset with configuration override."""
        env = CustomMiniGridEnv.from_preset(
            "empty_8x8", 
            max_steps=50
        )
        
        assert env.max_steps == 50
        env.close()
        
    def test_invalid_preset(self):
        """Test invalid preset raises error."""
        with pytest.raises(ValueError):
            CustomMiniGridEnv.from_preset("nonexistent_preset")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
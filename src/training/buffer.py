"""Experience replay buffers for training agents."""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Transition:
    """Single environment transition."""
    observation: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_observation: np.ndarray
    done: bool
    info: Optional[Dict] = None


class ReplayBuffer:
    """
    Standard replay buffer for off-policy learning.
    
    Stores individual transitions and samples random batches.
    Suitable for Actor-Critic and other model-free methods.
    """
    
    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        discrete_actions: bool = True
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            observation_shape: Shape of observations
            action_dim: Action dimension
            discrete_actions: Whether actions are discrete
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        
        # Flatten observation shape for storage
        self.flat_obs_dim = int(np.prod(observation_shape))
        
        # Pre-allocate arrays for efficiency
        self.observations = np.zeros(
            (capacity, self.flat_obs_dim), 
            dtype=np.float32
        )
        self.next_observations = np.zeros(
            (capacity, self.flat_obs_dim),
            dtype=np.float32
        )
        
        if discrete_actions:
            self.actions = np.zeros(capacity, dtype=np.int64)
        else:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
            
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
        
    def add(
        self,
        observation: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ) -> None:
        """
        Add transition to buffer.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
        """
        # Flatten observations
        self.observations[self.position] = observation.flatten()
        self.next_observations[self.position] = next_observation.flatten()
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(
        self,
        batch_size: int,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Sample random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            device: Torch device for tensors
            
        Returns:
            Dictionary of batched tensors
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "observations": torch.tensor(
                self.observations[indices], 
                dtype=torch.float32,
                device=device
            ),
            "actions": torch.tensor(
                self.actions[indices],
                dtype=torch.long if self.discrete_actions else torch.float32,
                device=device
            ),
            "rewards": torch.tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=device
            ),
            "next_observations": torch.tensor(
                self.next_observations[indices],
                dtype=torch.float32,
                device=device
            ),
            "dones": torch.tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=device
            )
        }
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size


class SequenceReplayBuffer:
    """
    Replay buffer storing sequences for world model training.
    
    Stores complete episodes and samples fixed-length sequences.
    Required for RSSM and other sequence models.
    """
    
    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        sequence_length: int = 20,
        discrete_actions: bool = True
    ):
        """
        Initialize sequence replay buffer.
        
        Args:
            capacity: Maximum number of episodes to store
            observation_shape: Shape of observations
            action_dim: Action dimension
            sequence_length: Length of sequences to sample
            discrete_actions: Whether actions are discrete
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.flat_obs_dim = int(np.prod(observation_shape))
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.discrete_actions = discrete_actions
        
        # Store complete episodes
        self.episodes: List[Dict[str, np.ndarray]] = []
        
        # Current episode being collected
        self._current_episode: Dict[str, List] = self._new_episode()
        
        # Debug counters
        self._total_transitions = 0
        
    def _new_episode(self) -> Dict[str, List]:
        """Create empty episode storage."""
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": []
        }
    
    def add(
        self,
        observation: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        done: bool
    ) -> None:
        """
        Add transition to current episode.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            done: Whether episode ended
        """
        # Flatten observation
        flat_obs = observation.flatten().astype(np.float32)
        
        self._current_episode["observations"].append(flat_obs)
        self._current_episode["actions"].append(action)
        self._current_episode["rewards"].append(reward)
        self._current_episode["dones"].append(done)
        self._total_transitions += 1
        
        if done:
            self._store_episode()
            
    def _store_episode(self) -> None:
        """Store completed episode and start new one."""
        ep_len = len(self._current_episode["observations"])
        
        if ep_len > 0:
            episode = {
                "observations": np.array(
                    self._current_episode["observations"],
                    dtype=np.float32
                ),
                "actions": np.array(
                    self._current_episode["actions"],
                    dtype=np.int64 if self.discrete_actions else np.float32
                ),
                "rewards": np.array(
                    self._current_episode["rewards"],
                    dtype=np.float32
                ),
                "dones": np.array(
                    self._current_episode["dones"],
                    dtype=np.float32
                )
            }
            
            self.episodes.append(episode)
            
            # Remove oldest episodes if over capacity
            while len(self.episodes) > self.capacity:
                self.episodes.pop(0)
                
        self._current_episode = self._new_episode()
        
    def end_episode(self) -> None:
        """Manually end current episode (e.g., on truncation)."""
        if len(self._current_episode["observations"]) > 0:
            # Mark last transition as done
            if self._current_episode["dones"]:
                self._current_episode["dones"][-1] = True
            self._store_episode()
    
    def sample(
        self,
        batch_size: int,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Sample batch of sequences.
        
        Args:
            batch_size: Number of sequences to sample
            device: Torch device
            
        Returns:
            Dictionary with tensors of shape (batch, seq_len, ...)
        """
        # Filter episodes long enough
        valid_episodes = [
            ep for ep in self.episodes 
            if len(ep["observations"]) >= self.sequence_length
        ]
        
        if len(valid_episodes) == 0:
            # If no episodes are long enough, use shorter sequences from available episodes
            if len(self.episodes) > 0:
                max_available = max(len(ep["observations"]) for ep in self.episodes)
                if max_available >= 2:
                    # Temporarily reduce sequence length
                    temp_seq_len = min(self.sequence_length, max_available)
                    valid_episodes = [
                        ep for ep in self.episodes 
                        if len(ep["observations"]) >= temp_seq_len
                    ]
                    actual_seq_len = temp_seq_len
                else:
                    raise ValueError(
                        f"No usable episodes. Have {len(self.episodes)} episodes, "
                        f"max length {max_available}, need at least 2."
                    )
            else:
                raise ValueError("No episodes in buffer.")
        else:
            actual_seq_len = self.sequence_length
        
        # Adjust batch size if needed
        effective_batch_size = min(batch_size, len(valid_episodes))
        
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for _ in range(effective_batch_size):
            # Sample random episode
            ep_idx = np.random.randint(0, len(valid_episodes))
            episode = valid_episodes[ep_idx]
            
            # Sample random starting point
            max_start = len(episode["observations"]) - actual_seq_len
            start_idx = np.random.randint(0, max_start + 1)
            end_idx = start_idx + actual_seq_len
            
            batch_obs.append(episode["observations"][start_idx:end_idx])
            batch_actions.append(episode["actions"][start_idx:end_idx])
            batch_rewards.append(episode["rewards"][start_idx:end_idx])
            batch_dones.append(episode["dones"][start_idx:end_idx])
            
        return {
            "observations": torch.tensor(
                np.array(batch_obs),
                dtype=torch.float32,
                device=device
            ),
            "actions": torch.tensor(
                np.array(batch_actions),
                dtype=torch.long if self.discrete_actions else torch.float32,
                device=device
            ),
            "rewards": torch.tensor(
                np.array(batch_rewards),
                dtype=torch.float32,
                device=device
            ),
            "dones": torch.tensor(
                np.array(batch_dones),
                dtype=torch.float32,
                device=device
            )
        }
    
    def __len__(self) -> int:
        """Return number of stored episodes."""
        return len(self.episodes)
    
    def total_transitions(self) -> int:
        """Return total number of transitions across all episodes."""
        return sum(len(ep["observations"]) for ep in self.episodes)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough sequences to sample."""
        if len(self.episodes) == 0:
            return False
        # Need at least 1 episode with enough length
        return any(
            len(ep["observations"]) >= min(self.sequence_length, 10)
            for ep in self.episodes
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics for debugging."""
        if len(self.episodes) == 0:
            return {
                "num_episodes": 0,
                "total_transitions": 0,
                "min_episode_length": 0,
                "max_episode_length": 0,
                "valid_episodes": 0
            }
        
        lengths = [len(ep["observations"]) for ep in self.episodes]
        return {
            "num_episodes": len(self.episodes),
            "total_transitions": sum(lengths),
            "min_episode_length": min(lengths),
            "max_episode_length": max(lengths),
            "valid_episodes": sum(1 for l in lengths if l >= self.sequence_length)
        }
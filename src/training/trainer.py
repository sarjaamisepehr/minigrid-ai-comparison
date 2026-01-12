"""Unified training loop for all agents."""
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
import numpy as np
import torch

from ..environments.base_env import BaseEnvironment
from ..agents.base_agent import BaseAgent
from ..agents.model_free.actor_critic import ActorCriticAgent
from ..agents.active_inference.ai_agent import ActiveInferenceAgent
from .buffer import ReplayBuffer, SequenceReplayBuffer
from .callbacks import Callback


class Trainer:
    """
    Unified trainer for both Actor-Critic and Active Inference agents.
    
    Handles:
    - Environment interaction
    - Experience collection
    - Agent updates
    - Callback management
    """
    
    def __init__(
        self,
        env: BaseEnvironment,
        agent: BaseAgent,
        config: Dict[str, Any],
        callbacks: Optional[List[Callback]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            env: Environment instance
            agent: Agent instance
            config: Training configuration with keys:
                - total_episodes: Number of episodes to train
                - max_steps_per_episode: Maximum steps per episode
                - batch_size: Batch size for updates
                - buffer_capacity: Replay buffer capacity
                - sequence_length: Sequence length for world model
                - updates_per_episode: Number of updates per episode
                - warmup_episodes: Episodes before starting updates
            callbacks: List of training callbacks
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.callbacks = callbacks or []
        
        # Extract config
        self.total_episodes = config.get("total_episodes", 1000)
        self.max_steps = config.get("max_steps_per_episode", 100)
        self.batch_size = config.get("batch_size", 64)
        self.buffer_capacity = config.get("buffer_capacity", 100000)
        self.sequence_length = config.get("sequence_length", 50)
        self.updates_per_episode = config.get("updates_per_episode", 1)
        self.warmup_episodes = config.get("warmup_episodes", 10)
        
        # Determine agent type and setup buffer
        self.is_active_inference = isinstance(agent, ActiveInferenceAgent)
        self._setup_buffer()
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        self.total_updates = 0
        
        # Logging
        self.agent_name = config.get("agent_name", type(agent).__name__)
        self.log_dir = config.get("log_dir", "logs")
        
    def _setup_buffer(self) -> None:
        """Setup appropriate replay buffer based on agent type."""
        obs_shape = self.env.get_observation_shape()
        action_dim = self.env.get_action_dim()
        discrete = self.env.is_discrete()
        
        if self.is_active_inference:
            # Sequence buffer for world model training
            self.buffer = SequenceReplayBuffer(
                capacity=self.buffer_capacity,
                observation_shape=obs_shape,
                action_dim=action_dim,
                sequence_length=self.sequence_length,
                discrete_actions=discrete
            )
        else:
            # Standard buffer for Actor-Critic
            self.buffer = ReplayBuffer(
                capacity=self.buffer_capacity,
                observation_shape=obs_shape,
                action_dim=action_dim,
                discrete_actions=discrete
            )
            
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Returns:
            Dictionary of training statistics
        """
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_training_start(self)
            
        try:
            for episode in range(1, self.total_episodes + 1):
                self.current_episode = episode
                episode_metrics = self._run_episode()
                
                # Notify callbacks
                for callback in self.callbacks:
                    callback.on_episode_end(
                        self,
                        episode,
                        episode_metrics["episode_reward"],
                        episode_metrics["episode_length"],
                        episode_metrics
                    )
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            
        finally:
            # Notify callbacks
            for callback in self.callbacks:
                callback.on_training_end(self)
                
        return self._get_training_stats()
    
    def _run_episode(self) -> Dict[str, float]:
        """
        Run single training episode.
        
        Returns:
            Episode metrics
        """
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_episode_start(self, self.current_episode)
            
        obs, info = self.env.reset()
        
        # Reset agent state if needed
        if hasattr(self.agent, 'reset_belief'):
            self.agent.reset_belief()
            
        episode_reward = 0.0
        episode_length = 0
        update_metrics = {}
        
        done = False
        
        while not done and episode_length < self.max_steps:
            # Select action
            output = self.agent.act(obs)
            action = output.action
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self._store_transition(obs, action, reward, next_obs, done, output)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            obs = next_obs
            
            # Notify callbacks
            for callback in self.callbacks:
                callback.on_step(self, self.total_steps, {})
                
        # End of episode - finalize buffer
        if self.is_active_inference:
            self.buffer.end_episode()
            
        # Perform updates
        if self.current_episode > self.warmup_episodes:
            update_metrics = self._perform_updates()
            
        # Aggregate metrics
        metrics = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            **update_metrics
        }
        
        self.agent.total_episodes += 1
        
        return metrics
    
    def _store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        output: Any
    ) -> None:
        """Store transition in appropriate buffer."""
        if self.is_active_inference:
            self.buffer.add(obs, action, reward, done)
        else:
            # For Actor-Critic, also store in agent's internal buffer
            if hasattr(self.agent, 'store_transition'):
                self.agent.store_transition(
                    observation=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    value=output.info.get("value", 0.0) if output.info else 0.0,
                    log_prob=output.log_prob
                )
            
            self.buffer.add(obs, action, reward, next_obs, done)
            
    def _perform_updates(self) -> Dict[str, float]:
        """
        Perform agent updates.
        
        Returns:
            Update metrics
        """
        all_metrics = {}
        
        if self.is_active_inference:
            # World model updates from buffer
            if self.buffer.is_ready(self.batch_size):
                for _ in range(self.updates_per_episode):
                    batch = self.buffer.sample(
                        self.batch_size,
                        device=str(self.agent.device)
                    )
                    metrics = self.agent.learn(batch)
                    
                    self.total_updates += 1
                    
                    # Accumulate metrics
                    for key, value in metrics.items():
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(value)
                        
                    # Notify callbacks
                    for callback in self.callbacks:
                        callback.on_update(self, self.total_updates, metrics)
        else:
            # Actor-Critic n-step updates
            if isinstance(self.agent, ActorCriticAgent):
                if self.agent.should_update():
                    # Get current observation for bootstrapping
                    metrics = self.agent.learn(next_observation=None)
                    
                    if metrics:
                        self.total_updates += 1
                        
                        for key, value in metrics.items():
                            if key not in all_metrics:
                                all_metrics[key] = []
                            all_metrics[key].append(value)
                            
                        for callback in self.callbacks:
                            callback.on_update(self, self.total_updates, metrics)
                            
        # Average metrics
        averaged_metrics = {
            key: np.mean(values) 
            for key, values in all_metrics.items()
        }
        
        return averaged_metrics
    
    def _get_training_stats(self) -> Dict[str, Any]:
        """Get final training statistics."""
        return {
            "total_episodes": self.current_episode,
            "total_steps": self.total_steps,
            "total_updates": self.total_updates,
            "agent_stats": self.agent.get_stats()
        }


def create_trainer(
    env: BaseEnvironment,
    agent_type: str,
    agent_config: Dict[str, Any],
    training_config: Dict[str, Any],
    callbacks: Optional[List[Callback]] = None,
    device: str = "cpu"
) -> Trainer:
    """
    Factory function to create trainer with specified agent.
    
    Args:
        env: Environment instance
        agent_type: "actor_critic" or "active_inference"
        agent_config: Agent configuration
        training_config: Training configuration
        callbacks: Training callbacks
        device: Torch device
        
    Returns:
        Configured Trainer instance
    """
    obs_shape = env.get_observation_shape()
    action_dim = env.get_action_dim()
    
    if agent_type == "actor_critic":
        agent = ActorCriticAgent(
            observation_shape=obs_shape,
            action_dim=action_dim,
            config=agent_config,
            device=device
        )
    elif agent_type == "active_inference":
        agent = ActiveInferenceAgent(
            observation_shape=obs_shape,
            action_dim=action_dim,
            config=agent_config,
            device=device
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    training_config["agent_name"] = agent_type
    
    return Trainer(
        env=env,
        agent=agent,
        config=training_config,
        callbacks=callbacks
    )
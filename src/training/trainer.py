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
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.callbacks = callbacks or []
        
        # Extract config
        self.total_episodes = config.get("total_episodes", 1000)
        self.max_steps = config.get("max_steps_per_episode", 100)
        self.batch_size = config.get("batch_size", 16)
        self.buffer_capacity = config.get("buffer_capacity", 10000)
        self.sequence_length = config.get("sequence_length", 20)
        self.updates_per_episode = config.get("updates_per_episode", 1)
        self.warmup_episodes = config.get("warmup_episodes", 5)
        
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
        
        print(f"Trainer initialized:")
        print(f"  Agent type: {'Active Inference' if self.is_active_inference else 'Actor-Critic'}")
        print(f"  Buffer type: {'Sequence' if self.is_active_inference else 'Standard'}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Warmup episodes: {self.warmup_episodes}")
        
    def _setup_buffer(self) -> None:
        """Setup appropriate replay buffer based on agent type."""
        obs_shape = self.env.get_observation_shape()
        action_dim = self.env.get_action_dim()
        discrete = self.env.is_discrete()
        
        if self.is_active_inference:
            self.buffer = SequenceReplayBuffer(
                capacity=self.buffer_capacity,
                observation_shape=obs_shape,
                action_dim=action_dim,
                sequence_length=self.sequence_length,
                discrete_actions=discrete
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=self.buffer_capacity,
                observation_shape=obs_shape,
                action_dim=action_dim,
                discrete_actions=discrete
            )
            
    def train(self) -> Dict[str, Any]:
        """Run training loop."""
        for callback in self.callbacks:
            callback.on_training_start(self)
            
        try:
            for episode in range(1, self.total_episodes + 1):
                self.current_episode = episode
                episode_metrics = self._run_episode()
                
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
            
        except Exception as e:
            print(f"\nTraining error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            for callback in self.callbacks:
                callback.on_training_end(self)
                
        return self._get_training_stats()
    
    def _run_episode(self) -> Dict[str, float]:
        """Run single training episode."""
        for callback in self.callbacks:
            callback.on_episode_start(self, self.current_episode)
            
        obs, info = self.env.reset()
        
        # Reset agent state if needed (Active Inference)
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
            
            # Store transition based on agent type
            if self.is_active_inference:
                self.buffer.add(obs, action, reward, done)
            else:
                # Store in replay buffer
                self.buffer.add(obs, action, reward, next_obs, done)
                
                # N-step update for Actor-Critic
                if hasattr(self.agent, 'store_transition'):
                    value = output.info.get("value", 0.0) if output.info else 0.0
                    log_prob = output.log_prob if output.log_prob is not None else torch.tensor(0.0)
                    
                    self.agent.store_transition(
                        observation=obs,
                        action=action,
                        reward=reward,
                        done=done,
                        value=value,
                        log_prob=log_prob
                    )
                    
                    if self.agent.should_update():
                        metrics = self.agent.learn(next_observation=next_obs)
                        if metrics:
                            self.total_updates += 1
                            for key, value in metrics.items():
                                if key not in update_metrics:
                                    update_metrics[key] = []
                                update_metrics[key].append(value)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            obs = next_obs
            
            for callback in self.callbacks:
                callback.on_step(self, self.total_steps, {})
                
        # End of episode
        if self.is_active_inference:
            # Finalize episode in buffer
            self.buffer.end_episode()
            
            # Perform world model updates after warmup
            if self.current_episode > self.warmup_episodes:
                wm_metrics = self._perform_world_model_updates()
                update_metrics.update(wm_metrics)
                
        # Average metrics
        averaged_metrics = {}
        for key, values in update_metrics.items():
            if isinstance(values, list):
                averaged_metrics[key] = np.mean(values)
            else:
                averaged_metrics[key] = values
        
        metrics = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            **averaged_metrics
        }
        
        self.agent.total_episodes += 1
        
        return metrics
    
    def _perform_world_model_updates(self) -> Dict[str, float]:
        """Perform world model updates for Active Inference."""
        all_metrics = {}
        
        # Check buffer status
        buffer_stats = self.buffer.get_stats()
        
        if buffer_stats["num_episodes"] < 1:
            return all_metrics
            
        # Try to sample and update
        try:
            for update_idx in range(self.updates_per_episode):
                batch = self.buffer.sample(
                    self.batch_size,
                    device=str(self.agent.device)
                )
                metrics = self.agent.learn(batch)
                
                self.total_updates += 1
                
                for key, value in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
                    
                for callback in self.callbacks:
                    callback.on_update(self, self.total_updates, metrics)
                    
        except Exception as e:
            if self.current_episode % 50 == 0:
                print(f"  [Debug] World model update issue: {e}")
                print(f"  [Debug] Buffer stats: {buffer_stats}")
                
        return all_metrics
    
    def _get_training_stats(self) -> Dict[str, Any]:
        """Get final training statistics."""
        return {
            "total_episodes": self.current_episode,
            "total_steps": self.total_steps,
            "total_updates": self.total_updates,
            "agent_stats": self.agent.get_stats()
        }
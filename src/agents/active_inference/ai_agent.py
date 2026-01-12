"""Active Inference agent using RSSM world model and EFE planning."""
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from ..base_agent import BaseAgent, AgentOutput
from .world_model import RSSMWorldModel
from .efe_planner import EFEPlanner, PolicyNetwork


class ActiveInferenceAgent(BaseAgent):
    """
    Active Inference agent combining:
    - RSSM world model for learning environment dynamics
    - EFE planner for action selection
    - Optional learned policy for faster inference
    
    Training consists of:
    1. World model learning from experience
    2. Policy learning to match EFE-optimal actions (optional)
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """
        Initialize Active Inference agent.
        
        Args:
            observation_shape: Shape of observations
            action_dim: Number of discrete actions
            config: Configuration with keys:
                - world_model.*: RSSM configuration
                - planning.*: EFE planner configuration
                - learning_rate: Optimizer learning rate
                - gamma: Discount factor
                - beta_kl: KL divergence weight
        """
        super().__init__(observation_shape, action_dim, config, device)
        
        self.observation_dim = int(np.prod(observation_shape))
        
        # World model config
        wm_config = config.get("world_model", {})
        self.deterministic_dim = wm_config.get("deterministic_dim", 200)
        self.stochastic_dim = wm_config.get("stochastic_dim", 30)
        self.hidden_dim = wm_config.get("hidden_dim", 200)
        self.embedding_dim = wm_config.get("embedding_dim", 200)
        
        # Planning config
        plan_config = config.get("planning", {})
        self.horizon = plan_config.get("horizon", 10)
        self.num_samples = plan_config.get("num_samples", 50)
        self.temperature = plan_config.get("temperature", 1.0)
        self.epistemic_weight = plan_config.get("epistemic_weight", 1.0)
        
        # Training config
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.beta_kl = config.get("beta_kl", 1.0)
        self.free_nats = config.get("free_nats", 3.0)
        
        # Build world model
        self.world_model = RSSMWorldModel(
            observation_dim=self.observation_dim,
            action_dim=action_dim,
            deterministic_dim=self.deterministic_dim,
            stochastic_dim=self.stochastic_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Build planner
        self.planner = EFEPlanner(
            world_model=self.world_model,
            action_dim=action_dim,
            horizon=self.horizon,
            num_samples=self.num_samples,
            temperature=self.temperature,
            gamma=self.gamma,
            epistemic_weight=self.epistemic_weight
        )
        
        # Optional learned policy for faster inference
        state_dim = self.deterministic_dim + self.stochastic_dim
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.world_model_optimizer = optim.Adam(
            self.world_model.parameters(),
            lr=self.lr
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.lr
        )
        
        # Current belief state (updated during episode)
        self._current_state: Optional[Dict[str, torch.Tensor]] = None
        self._prev_action: Optional[torch.Tensor] = None
        
        # Whether to use learned policy or EFE planning
        self.use_learned_policy = config.get("use_learned_policy", False)
        
    def reset_belief(self):
        """Reset belief state at start of episode."""
        self._current_state = self.world_model.initial_state(1, self.device)
        self._prev_action = torch.zeros(1, dtype=torch.long, device=self.device)
        
    def update_belief(self, observation: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Update belief state given new observation.
        
        Args:
            observation: Current observation
            
        Returns:
            Updated state dict
        """
        if self._current_state is None:
            self.reset_belief()
            
        obs_tensor = self._to_tensor(observation).view(1, -1)
        
        self._current_state, _ = self.world_model.observe_step(
            self._current_state,
            self._prev_action,
            obs_tensor
        )
        
        return self._current_state
    
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
        # Update belief with new observation
        state = self.update_belief(observation)
        
        if self.use_learned_policy and not self.training:
            # Use learned policy for fast inference
            state_features = self.world_model.get_features(state)
            action, log_prob = self.policy.get_action(
                state_features,
                deterministic=deterministic
            )
            action = action.item()
            info = {"method": "learned_policy"}
        else:
            # Use EFE planning
            action, plan_info = self.planner.plan(
                state,
                deterministic=deterministic
            )
            info = {"method": "efe_planning", **plan_info}
            
        # Store action for next belief update
        self._prev_action = torch.tensor([action], device=self.device)
        
        # Get predicted value (expected reward)
        with torch.no_grad():
            predicted_reward = self.world_model.predict_reward(state)
            
        return AgentOutput(
            action=action,
            value=predicted_reward,
            info=info
        )
    
    def learn(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Update agent from batch of experience.
        
        Args:
            batch: Dictionary with:
                - observations: (batch, seq_len, obs_dim)
                - actions: (batch, seq_len)
                - rewards: (batch, seq_len)
                - dones: (batch, seq_len)
                
        Returns:
            Dictionary of loss values
        """
        # Ensure correct shapes
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Ensure sequence dimension exists
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)
            actions = actions.unsqueeze(1)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)
            
        # World model learning
        wm_losses = self.world_model.compute_loss(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            kl_weight=self.beta_kl,
            free_nats=self.free_nats
        )
        
        self.world_model_optimizer.zero_grad()
        wm_losses["total_loss"].backward()
        self.world_model_optimizer.step()
        
        # Policy learning (distillation from EFE planner)
        policy_loss = self._update_policy(observations, actions)
        
        # Update statistics
        self.total_steps += observations.shape[0] * observations.shape[1]
        
        metrics = {
            "world_model_loss": wm_losses["total_loss"].item(),
            "observation_loss": wm_losses["observation_loss"].item(),
            "reward_loss": wm_losses["reward_loss"].item(),
            "kl_loss": wm_losses["kl_loss"].item(),
            "policy_loss": policy_loss
        }
        
        return metrics
    
    def _update_policy(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> float:
        """
        Update policy to match world model predictions.
        
        Simple behavioral cloning on observed actions.
        Could be extended to match EFE-optimal actions.
        """
        batch_size, seq_len, _ = observations.shape
        
        # Get state features for all observations
        state = self.world_model.initial_state(batch_size, self.device)
        prev_action = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        total_loss = 0.0
        
        for t in range(seq_len):
            obs_t = observations[:, t]
            action_t = actions[:, t]
            
            # Update state
            state, _ = self.world_model.observe_step(state, prev_action, obs_t)
            
            # Get policy prediction
            state_features = self.world_model.get_features(state)
            logits = self.policy(state_features)
            
            # Cross-entropy loss (behavioral cloning)
            loss = torch.nn.functional.cross_entropy(logits, action_t)
            total_loss += loss
            
            prev_action = action_t
            
        total_loss = total_loss / seq_len
        
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        
        return total_loss.item()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save agent state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "world_model_state_dict": self.world_model.state_dict(),
            "policy_state_dict": self.policy.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "config": self.config,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes
        }, path)
        
    def load(self, path: Union[str, Path]) -> None:
        """Load agent state."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint["world_model_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.world_model_optimizer.load_state_dict(checkpoint["world_model_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.total_episodes = checkpoint.get("total_episodes", 0)
        
    def train_mode(self) -> None:
        """Set to training mode."""
        super().train_mode()
        self.world_model.train()
        self.policy.train()
        
    def eval_mode(self) -> None:
        """Set to evaluation mode."""
        super().eval_mode()
        self.world_model.eval()
        self.policy.eval()
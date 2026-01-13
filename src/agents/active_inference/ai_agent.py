"""Active Inference agent using RSSM world model - Simplified Version."""
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from ..base_agent import BaseAgent, AgentOutput
from .world_model import RSSMWorldModel
from .efe_planner import EFEPlanner, RandomPlanner


class ActiveInferenceAgent(BaseAgent):
    """
    Simplified Active Inference agent.
    
    Uses RSSM world model with simple 1-step EFE planning.
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """Initialize Active Inference agent."""
        super().__init__(observation_shape, action_dim, config, device)
        
        self.observation_dim = int(np.prod(observation_shape))
        
        # World model config
        wm_config = config.get("world_model", {})
        self.deterministic_dim = wm_config.get("deterministic_dim", 64)
        self.stochastic_dim = wm_config.get("stochastic_dim", 16)
        self.hidden_dim = wm_config.get("hidden_dim", 64)
        self.embedding_dim = wm_config.get("embedding_dim", 64)
        
        # Planning config
        plan_config = config.get("planning", {})
        self.temperature = plan_config.get("temperature", 1.0)
        self.epistemic_weight = plan_config.get("epistemic_weight", 0.1)
        
        # Training config
        self.lr = config.get("learning_rate", 1e-3)
        self.gamma = config.get("gamma", 0.99)
        self.beta_kl = config.get("beta_kl", 0.1)
        self.free_nats = config.get("free_nats", 1.0)
        
        # Build world model
        print(f"  Building world model: obs_dim={self.observation_dim}, act_dim={action_dim}")
        self.world_model = RSSMWorldModel(
            observation_dim=self.observation_dim,
            action_dim=action_dim,
            deterministic_dim=self.deterministic_dim,
            stochastic_dim=self.stochastic_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Build planner (simple 1-step)
        self.planner = EFEPlanner(
            world_model=self.world_model,
            action_dim=action_dim,
            horizon=1,
            temperature=self.temperature,
            epistemic_weight=self.epistemic_weight
        )
        
        # Use random planner initially until world model is trained
        self.random_planner = RandomPlanner(action_dim)
        self.use_random = True  # Start with random actions
        self.warmup_steps = 500  # Use random for first N steps
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.world_model.parameters(),
            lr=self.lr
        )
        
        # Current belief state
        self._current_state: Optional[Dict[str, torch.Tensor]] = None
        self._prev_action: Optional[torch.Tensor] = None
        self._step_count = 0
        
        print(f"  Active Inference agent initialized")
        
    def reset_belief(self):
        """Reset belief state at start of episode."""
        self._current_state = self.world_model.initial_state(1, self.device)
        self._prev_action = torch.zeros(1, dtype=torch.long, device=self.device)
        
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> AgentOutput:
        """Select action given observation."""
        self._step_count += 1
        
        # Flatten observation
        obs_flat = observation.flatten().astype(np.float32)
        obs_tensor = torch.tensor(obs_flat, device=self.device).unsqueeze(0)
        
        # Initialize belief if needed
        if self._current_state is None:
            self.reset_belief()
            
        # Update belief with observation
        try:
            self._current_state, _ = self.world_model.observe_step(
                self._current_state,
                self._prev_action,
                obs_tensor
            )
        except Exception as e:
            print(f"Warning: Belief update failed: {e}")
            self.reset_belief()
        
        # Select action
        if self.use_random and self._step_count < self.warmup_steps:
            # Random exploration during warmup
            action = np.random.randint(0, self.action_dim)
            info = {"method": "random_warmup"}
        else:
            # Use EFE planner
            self.use_random = False
            try:
                action, info = self.planner.plan(
                    self._current_state,
                    deterministic=deterministic
                )
            except Exception as e:
                print(f"Warning: Planning failed: {e}")
                action = np.random.randint(0, self.action_dim)
                info = {"method": "random_fallback"}
        
        # Store action for next belief update
        self._prev_action = torch.tensor([action], device=self.device)
        
        # Get predicted reward as value estimate
        try:
            with torch.no_grad():
                value = self.world_model.predict_reward(self._current_state).item()
        except:
            value = 0.0
            
        return AgentOutput(
            action=action,
            value=torch.tensor(value),
            info=info
        )
    
    def learn(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Update world model from batch of experience."""
        try:
            observations = batch["observations"].to(self.device)
            actions = batch["actions"].to(self.device)
            rewards = batch["rewards"].to(self.device)
            dones = batch["dones"].to(self.device)
            
            # Ensure sequence dimension
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
            
            self.optimizer.zero_grad()
            wm_losses["total_loss"].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
            
            self.optimizer.step()
            
            self.total_steps += observations.shape[0] * observations.shape[1]
            
            return {
                "world_model_loss": wm_losses["total_loss"].item(),
                "observation_loss": wm_losses["observation_loss"].item(),
                "reward_loss": wm_losses["reward_loss"].item(),
                "kl_loss": wm_losses["kl_loss"].item()
            }
            
        except Exception as e:
            print(f"Warning: Learning failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "world_model_loss": 0.0,
                "observation_loss": 0.0,
                "reward_loss": 0.0,
                "kl_loss": 0.0
            }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save agent state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "world_model_state_dict": self.world_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes
        }, path)
        
    def load(self, path: Union[str, Path]) -> None:
        """Load agent state."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint["world_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.total_episodes = checkpoint.get("total_episodes", 0)
        self.use_random = False  # Disable random after loading
        
    def train_mode(self) -> None:
        """Set to training mode."""
        super().train_mode()
        self.world_model.train()
        
    def eval_mode(self) -> None:
        """Set to evaluation mode."""
        super().eval_mode()
        self.world_model.eval()
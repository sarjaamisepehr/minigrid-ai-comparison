"""
Expected Free Energy (EFE) planner for Active Inference.

EFE combines:
- Pragmatic value: Expected reward/utility
- Epistemic value: Expected information gain (uncertainty reduction)
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .world_model import RSSMWorldModel


class EFEPlanner(nn.Module):
    """
    Expected Free Energy planner for action selection.
    
    Evaluates actions based on:
    G(a) = E[risk] + E[ambiguity] - E[info_gain]
    
    Where lower EFE is better (minimizing free energy).
    """
    
    def __init__(
        self,
        world_model: RSSMWorldModel,
        action_dim: int,
        horizon: int = 10,
        num_samples: int = 50,
        temperature: float = 1.0,
        gamma: float = 0.99,
        epistemic_weight: float = 1.0
    ):
        """
        Initialize EFE planner.
        
        Args:
            world_model: Trained RSSM world model
            action_dim: Number of discrete actions
            horizon: Planning horizon
            num_samples: Number of trajectory samples per action
            temperature: Softmax temperature for action selection
            gamma: Discount factor for future rewards
            epistemic_weight: Weight for epistemic value (exploration bonus)
        """
        super().__init__()
        
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.gamma = gamma
        self.epistemic_weight = epistemic_weight
        
    def compute_efe(
        self,
        state: Dict[str, torch.Tensor],
        action_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Expected Free Energy for action sequence.
        
        Args:
            state: Current state dict
            action_sequence: (horizon,) tensor of actions
            
        Returns:
            EFE value (lower is better)
        """
        device = state["h"].device
        batch_size = state["h"].shape[0]
        
        current_state = {
            "h": state["h"].clone(),
            "s": state["s"].clone()
        }
        
        total_pragmatic = torch.zeros(batch_size, device=device)
        total_epistemic = torch.zeros(batch_size, device=device)
        
        discount = 1.0
        
        for t in range(len(action_sequence)):
            action = action_sequence[t].expand(batch_size)
            
            # Imagine next state
            next_state = self.world_model.imagine_step(current_state, action)
            
            # Pragmatic value: Expected reward
            predicted_reward = self.world_model.predict_reward(next_state)
            total_pragmatic += discount * predicted_reward
            
            # Epistemic value: State uncertainty (entropy of posterior)
            # Higher uncertainty = more information to gain = higher epistemic value
            state_features = self.world_model.get_features(next_state)
            
            # Approximate epistemic value using prediction uncertainty
            # This is a simplification - could use more sophisticated measures
            obs_pred = self.world_model.decode_observation(next_state)
            
            # Uncertainty as variance in predictions (simplified)
            epistemic = torch.var(state_features, dim=-1)
            total_epistemic += discount * epistemic
            
            current_state = next_state
            discount *= self.gamma
            
        # EFE = -pragmatic_value - epistemic_weight * epistemic_value
        # (Negative because we want to maximize reward and info gain)
        efe = -total_pragmatic - self.epistemic_weight * total_epistemic
        
        return efe
    
    def plan(
        self,
        state: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[int, Dict[str, float]]:
        """
        Select action by evaluating EFE for all actions.
        
        Uses simple one-step lookahead for efficiency.
        For longer horizons, samples random action sequences.
        
        Args:
            state: Current state dict
            deterministic: If True, return argmin action
            
        Returns:
            action: Selected action
            info: Planning information
        """
        device = state["h"].device
        
        # Evaluate each first action
        efe_values = []
        
        for action in range(self.action_dim):
            if self.horizon == 1:
                # Simple one-step lookahead
                action_tensor = torch.tensor([action], device=device)
                efe = self.compute_efe(state, action_tensor.unsqueeze(0))
            else:
                # Monte Carlo sampling for longer horizons
                total_efe = torch.zeros(1, device=device)
                
                for _ in range(self.num_samples):
                    # Random action sequence starting with current action
                    action_seq = torch.randint(
                        0, self.action_dim, 
                        (self.horizon,), 
                        device=device
                    )
                    action_seq[0] = action
                    
                    efe = self.compute_efe(state, action_seq)
                    total_efe += efe
                    
                efe = total_efe / self.num_samples
                
            efe_values.append(efe.item())
            
        efe_tensor = torch.tensor(efe_values, device=device)
        
        # Convert EFE to action probabilities (lower EFE = higher prob)
        # Note: negative because lower EFE is better
        action_probs = F.softmax(-efe_tensor / self.temperature, dim=0)
        
        if deterministic:
            action = efe_tensor.argmin().item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            
        info = {
            "efe_values": efe_values,
            "action_probs": action_probs.cpu().numpy().tolist(),
            "min_efe": min(efe_values),
            "max_efe": max(efe_values)
        }
        
        return action, info


class PolicyNetwork(nn.Module):
    """
    Learned policy network for Active Inference.
    
    Can be trained via policy gradient to match EFE-optimal actions,
    providing faster action selection at deployment.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state features
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """Compute action logits."""
        return self.network(state_features)
    
    def get_action(
        self,
        state_features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: Selected action
            log_prob: Log probability
        """
        logits = self.forward(state_features)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        
        return action, log_prob
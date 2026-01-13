"""
Expected Free Energy (EFE) planner for Active Inference - Simplified Version.
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .world_model import RSSMWorldModel


class EFEPlanner(nn.Module):
    """
    Simplified Expected Free Energy planner for action selection.
    
    Uses 1-step lookahead for speed while maintaining exploration.
    """
    
    def __init__(
        self,
        world_model: RSSMWorldModel,
        action_dim: int,
        horizon: int = 1,  # Default to 1-step for speed
        num_samples: int = 1,
        temperature: float = 1.0,
        gamma: float = 0.99,
        epistemic_weight: float = 1.0
    ):
        """
        Initialize EFE planner.
        
        Args:
            world_model: Trained RSSM world model
            action_dim: Number of discrete actions
            horizon: Planning horizon (1 for fast planning)
            num_samples: Number of trajectory samples (1 for fast planning)
            temperature: Softmax temperature for action selection
            gamma: Discount factor
            epistemic_weight: Weight for epistemic value (exploration)
        """
        super().__init__()
        
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.gamma = gamma
        self.epistemic_weight = epistemic_weight
        
    def compute_action_values(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute value for each action using 1-step lookahead.
        
        Args:
            state: Current state dict with 'h' and 's'
            
        Returns:
            Tensor of action values (higher is better)
        """
        device = state["h"].device
        action_values = []
        
        with torch.no_grad():
            for action in range(self.action_dim):
                action_tensor = torch.tensor([action], device=device)
                
                # Imagine next state
                next_state = self.world_model.imagine_step(state, action_tensor)
                
                # Pragmatic value: Expected reward
                predicted_reward = self.world_model.predict_reward(next_state)
                
                # Epistemic value: Uncertainty in state (simplified)
                # Use variance of stochastic state as proxy for uncertainty
                epistemic_value = torch.var(next_state["s"])
                
                # Combined value (higher is better)
                value = predicted_reward + self.epistemic_weight * epistemic_value
                action_values.append(value.item())
                
        return torch.tensor(action_values, device=device)
    
    def plan(
        self,
        state: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[int, Dict[str, float]]:
        """
        Select action based on Expected Free Energy.
        
        Args:
            state: Current state dict
            deterministic: If True, return argmax action
            
        Returns:
            action: Selected action
            info: Planning information
        """
        # Get action values
        action_values = self.compute_action_values(state)
        
        # Convert to probabilities
        action_probs = F.softmax(action_values / self.temperature, dim=0)
        
        if deterministic:
            action = action_values.argmax().item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            
        info = {
            "action_values": action_values.cpu().numpy().tolist(),
            "action_probs": action_probs.cpu().numpy().tolist(),
            "max_value": action_values.max().item(),
            "min_value": action_values.min().item()
        }
        
        return action, info


class RandomPlanner:
    """Simple random action selection for debugging."""
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        
    def plan(
        self,
        state: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[int, Dict[str, float]]:
        """Select random action."""
        import random
        action = random.randint(0, self.action_dim - 1)
        return action, {"method": "random"}
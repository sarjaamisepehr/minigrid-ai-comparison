"""
Recurrent State-Space Model (RSSM) for Active Inference.

The RSSM learns a world model with:
- Deterministic path: GRU-based recurrent state
- Stochastic path: Learned latent distribution
- Observation model: Decodes observations from state
- Reward model: Predicts rewards from state
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class RSSMWorldModel(nn.Module):
    """
    Recurrent State-Space Model for learning environment dynamics.
    
    State consists of:
    - h: Deterministic recurrent state (GRU hidden)
    - s: Stochastic latent state (sampled from learned distribution)
    
    Models:
    - Transition: p(s_t | h_t)
    - Posterior: q(s_t | h_t, o_t) 
    - Observation: p(o_t | h_t, s_t)
    - Reward: p(r_t | h_t, s_t)
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        deterministic_dim: int = 200,
        stochastic_dim: int = 30,
        hidden_dim: int = 200,
        embedding_dim: int = 200,
        min_std: float = 0.1
    ):
        """
        Initialize RSSM.
        
        Args:
            observation_dim: Dimension of observations
            action_dim: Number of discrete actions
            deterministic_dim: Size of deterministic state h
            stochastic_dim: Size of stochastic state s
            hidden_dim: Size of hidden layers in MLPs
            embedding_dim: Size of observation embedding
            min_std: Minimum standard deviation for distributions
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std
        
        # State dimension
        self.state_dim = deterministic_dim + stochastic_dim
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ELU()
        )
        
        # Action embedding (for discrete actions)
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        
        # Recurrent model: GRU for deterministic path
        # Input: previous stochastic state + action embedding
        self.rnn = nn.GRUCell(
            input_size=stochastic_dim + hidden_dim,
            hidden_size=deterministic_dim
        )
        
        # Prior network: p(s_t | h_t)
        # Predicts stochastic state from deterministic state alone
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stochastic_dim)  # mean and std
        )
        
        # Posterior network: q(s_t | h_t, o_t)
        # Predicts stochastic state from deterministic state + observation
        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_dim + embedding_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stochastic_dim)  # mean and std
        )
        
        # Observation decoder: p(o_t | h_t, s_t)
        self.obs_decoder = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, observation_dim)
        )
        
        # Reward predictor: p(r_t | h_t, s_t)
        self.reward_predictor = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Continue predictor (optional): p(continue | h_t, s_t)
        self.continue_predictor = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Get initial state for beginning of episode.
        
        Args:
            batch_size: Batch size
            device: Torch device
            
        Returns:
            Dictionary with 'h' (deterministic) and 's' (stochastic) states
        """
        return {
            "h": torch.zeros(batch_size, self.deterministic_dim, device=device),
            "s": torch.zeros(batch_size, self.stochastic_dim, device=device)
        }
        
    def _get_dist(self, mean_std: torch.Tensor) -> Normal:
        """Create Normal distribution from concatenated mean and std."""
        mean, std = torch.chunk(mean_std, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        return Normal(mean, std)
    
    def observe_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        observation: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Single step of observation-conditioned state update.
        
        Used during training when we have actual observations.
        
        Args:
            prev_state: Previous state dict with 'h' and 's'
            prev_action: Previous action (discrete indices)
            observation: Current observation
            
        Returns:
            state: New state dict
            distributions: Dict with 'prior' and 'posterior' distributions
        """
        # Embed action
        action_emb = self.action_embedding(prev_action)
        
        # Update deterministic state
        rnn_input = torch.cat([prev_state["s"], action_emb], dim=-1)
        h = self.rnn(rnn_input, prev_state["h"])
        
        # Compute prior p(s|h)
        prior_params = self.prior_net(h)
        prior_dist = self._get_dist(prior_params)
        
        # Encode observation
        obs_emb = self.obs_encoder(observation)
        
        # Compute posterior q(s|h,o)
        posterior_input = torch.cat([h, obs_emb], dim=-1)
        posterior_params = self.posterior_net(posterior_input)
        posterior_dist = self._get_dist(posterior_params)
        
        # Sample from posterior (training uses reparameterization)
        s = posterior_dist.rsample()
        
        state = {"h": h, "s": s}
        distributions = {
            "prior": prior_dist,
            "posterior": posterior_dist
        }
        
        return state, distributions
    
    def imagine_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Single step of imagination (no observation).
        
        Used during planning when we don't have real observations.
        
        Args:
            prev_state: Previous state dict
            action: Action to take
            
        Returns:
            New state dict
        """
        # Embed action
        action_emb = self.action_embedding(action)
        
        # Update deterministic state
        rnn_input = torch.cat([prev_state["s"], action_emb], dim=-1)
        h = self.rnn(rnn_input, prev_state["h"])
        
        # Sample from prior (no observation available)
        prior_params = self.prior_net(h)
        prior_dist = self._get_dist(prior_params)
        s = prior_dist.rsample()
        
        return {"h": h, "s": s}
    
    def decode_observation(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Decode observation from state."""
        state_features = torch.cat([state["h"], state["s"]], dim=-1)
        return self.obs_decoder(state_features)
    
    def predict_reward(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Predict reward from state."""
        state_features = torch.cat([state["h"], state["s"]], dim=-1)
        return self.reward_predictor(state_features).squeeze(-1)
    
    def predict_continue(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Predict continuation probability from state."""
        state_features = torch.cat([state["h"], state["s"]], dim=-1)
        return torch.sigmoid(self.continue_predictor(state_features)).squeeze(-1)
    
    def get_features(
        self,
        state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get concatenated state features for downstream use."""
        return torch.cat([state["h"], state["s"]], dim=-1)
    
    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        kl_weight: float = 1.0,
        free_nats: float = 3.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute world model losses over a sequence.
        
        Args:
            observations: (batch, seq_len, obs_dim)
            actions: (batch, seq_len) discrete actions
            rewards: (batch, seq_len)
            dones: (batch, seq_len)
            kl_weight: Weight for KL divergence loss
            free_nats: Free nats for KL (minimum KL)
            
        Returns:
            Dictionary of losses
        """
        batch_size, seq_len, _ = observations.shape
        device = observations.device
        
        # Initialize state
        state = self.initial_state(batch_size, device)
        
        # Storage for losses
        obs_losses = []
        reward_losses = []
        kl_losses = []
        
        # Assume first action is zero (or could be passed in)
        prev_action = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for t in range(seq_len):
            # Get current observation and action
            obs_t = observations[:, t]
            action_t = actions[:, t] if t < seq_len - 1 else prev_action
            
            # Update state with observation
            state, dists = self.observe_step(state, prev_action, obs_t)
            
            # Reconstruction loss
            obs_pred = self.decode_observation(state)
            obs_loss = F.mse_loss(obs_pred, obs_t, reduction='none').sum(-1)
            obs_losses.append(obs_loss)
            
            # Reward prediction loss
            reward_pred = self.predict_reward(state)
            reward_loss = F.mse_loss(reward_pred, rewards[:, t], reduction='none')
            reward_losses.append(reward_loss)
            
            # KL divergence loss
            kl = kl_divergence(dists["posterior"], dists["prior"]).sum(-1)
            kl = torch.max(kl, torch.tensor(free_nats, device=device))
            kl_losses.append(kl)
            
            prev_action = action_t
            
        # Aggregate losses
        obs_loss = torch.stack(obs_losses, dim=1).mean()
        reward_loss = torch.stack(reward_losses, dim=1).mean()
        kl_loss = torch.stack(kl_losses, dim=1).mean()
        
        total_loss = obs_loss + reward_loss + kl_weight * kl_loss
        
        return {
            "total_loss": total_loss,
            "observation_loss": obs_loss,
            "reward_loss": reward_loss,
            "kl_loss": kl_loss
        }
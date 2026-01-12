"""Tests for agent implementations."""
import pytest
import numpy as np
import torch

from src.agents import ActorCriticAgent, ActiveInferenceAgent
from src.agents.base_agent import AgentOutput


class TestActorCriticAgent:
    """Test suite for Actor-Critic agent."""
    
    @pytest.fixture
    def config(self):
        return {
            "network": {
                "hidden_dims": [64, 64],
                "activation": "relu"
            },
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_steps": 5
        }
    
    @pytest.fixture
    def agent(self, config):
        return ActorCriticAgent(
            observation_shape=(147,),
            action_dim=7,
            config=config,
            device="cpu"
        )
    
    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.action_dim == 7
        
    def test_act(self, agent):
        """Test action selection."""
        obs = np.random.randn(147).astype(np.float32)
        output = agent.act(obs)
        
        assert isinstance(output, AgentOutput)
        assert 0 <= output.action < 7
        assert output.value is not None
        
    def test_act_deterministic(self, agent):
        """Test deterministic action selection."""
        obs = np.random.randn(147).astype(np.float32)
        
        # Multiple deterministic calls should return same action
        actions = [agent.act(obs, deterministic=True).action for _ in range(5)]
        assert len(set(actions)) == 1
        
    def test_store_and_update(self, agent):
        """Test storing transitions and updating."""
        obs = np.random.randn(147).astype(np.float32)
        
        for _ in range(agent.n_steps):
            output = agent.act(obs)
            agent.store_transition(
                observation=obs,
                action=output.action,
                reward=1.0,
                done=False,
                value=output.info["value"],
                log_prob=output.log_prob
            )
            
        assert agent.should_update()
        
        next_obs = np.random.randn(147).astype(np.float32)
        metrics = agent.learn(next_observation=next_obs)
        
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        
    def test_save_load(self, agent, tmp_path):
        """Test saving and loading agent."""
        save_path = tmp_path / "agent.pt"
        
        # Store initial state
        obs = np.random.randn(147).astype(np.float32)
        initial_action = agent.act(obs, deterministic=True).action
        
        # Save
        agent.save(save_path)
        
        # Create new agent and load
        new_agent = ActorCriticAgent(
            observation_shape=(147,),
            action_dim=7,
            config=agent.config,
            device="cpu"
        )
        new_agent.load(save_path)
        
        # Should produce same action
        loaded_action = new_agent.act(obs, deterministic=True).action
        assert initial_action == loaded_action


class TestActiveInferenceAgent:
    """Test suite for Active Inference agent."""
    
    @pytest.fixture
    def config(self):
        return {
            "world_model": {
                "deterministic_dim": 64,
                "stochastic_dim": 16,
                "hidden_dim": 64,
                "embedding_dim": 64
            },
            "planning": {
                "horizon": 3,
                "num_samples": 10,
                "temperature": 1.0,
                "epistemic_weight": 1.0
            },
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "beta_kl": 1.0
        }
    
    @pytest.fixture
    def agent(self, config):
        return ActiveInferenceAgent(
            observation_shape=(147,),
            action_dim=7,
            config=config,
            device="cpu"
        )
    
    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.action_dim == 7
        assert agent.world_model is not None
        
    def test_reset_belief(self, agent):
        """Test belief state reset."""
        agent.reset_belief()
        assert agent._current_state is not None
        assert agent._prev_action is not None
        
    def test_act(self, agent):
        """Test action selection with EFE planning."""
        agent.reset_belief()
        obs = np.random.randn(147).astype(np.float32)
        output = agent.act(obs)
        
        assert isinstance(output, AgentOutput)
        assert 0 <= output.action < 7
        
    def test_learn(self, agent):
        """Test world model learning."""
        batch = {
            "observations": torch.randn(4, 10, 147),
            "actions": torch.randint(0, 7, (4, 10)),
            "rewards": torch.randn(4, 10),
            "dones": torch.zeros(4, 10)
        }
        
        metrics = agent.learn(batch)
        
        assert "world_model_loss" in metrics
        assert "observation_loss" in metrics
        assert "kl_loss" in metrics
        
    def test_save_load(self, agent, tmp_path):
        """Test saving and loading agent."""
        save_path = tmp_path / "ai_agent.pt"
        
        # Save
        agent.save(save_path)
        
        # Load
        new_agent = ActiveInferenceAgent(
            observation_shape=(147,),
            action_dim=7,
            config=agent.config,
            device="cpu"
        )
        new_agent.load(save_path)
        
        assert new_agent.total_steps == agent.total_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
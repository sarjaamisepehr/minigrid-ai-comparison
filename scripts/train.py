#!/usr/bin/env python
"""Main training script for single agent."""
import argparse
from pathlib import Path
import torch

from src.environments import MiniGridEnvironment
from src.agents import ActorCriticAgent, ActiveInferenceAgent
from src.training import Trainer, LoggingCallback, CheckpointCallback, EvaluationCallback
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train agent in MiniGrid")
    
    parser.add_argument(
        "--agent",
        type=str,
        choices=["actor_critic", "active_inference"],
        default="actor_critic",
        help="Agent type to train"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-Empty-5x5-v0",
        help="MiniGrid environment ID"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Logging directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    print("=" * 60)
    print(f"Training {args.agent.upper()} on {args.env}")
    print("=" * 60)
    
    # Environment config
    env_config = {
        "env_id": args.env,
        "observation_type": "flat",
        "fully_observable": False,
        "max_steps": 100,
        "render_mode": None
    }
    
    env = MiniGridEnvironment(env_config)
    obs_shape = env.get_observation_shape()
    action_dim = env.get_action_dim()
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {args.device}")
    
    # Agent configs
    if args.agent == "actor_critic":
        agent_config = {
            "network": {
                "hidden_dims": [64, 64],
                "activation": "relu"
            },
            "learning_rate": 7e-4,
            "gamma": 0.99,
            "entropy_coef": 0.1,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_steps": 8
        }
        
        agent = ActorCriticAgent(
            observation_shape=obs_shape,
            action_dim=action_dim,
            config=agent_config,
            device=args.device
        )
        
    else:  # active_inference
        agent_config = {
            "world_model": {
                "deterministic_dim": 64,
                "stochastic_dim": 16,
                "hidden_dim": 64,
                "embedding_dim": 64
            },
            "planning": {
                "temperature": 1.0,
                "epistemic_weight": 0.1
            },
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "beta_kl": 0.1,
            "free_nats": 1.0
        }
        
        agent = ActiveInferenceAgent(
            observation_shape=obs_shape,
            action_dim=action_dim,
            config=agent_config,
            device=args.device
        )
        
    # Training config
    training_config = {
        "total_episodes": args.episodes,
        "max_steps_per_episode": 100,
        "batch_size": 8,
        "buffer_capacity": 5000,
        "sequence_length": 15,
        "updates_per_episode": 2,
        "warmup_episodes": 3,
        "log_dir": args.log_dir,
        "agent_name": args.agent
    }
    
    # Callbacks - Note: deterministic=False for evaluation
    callbacks = [
        LoggingCallback(
            log_dir=args.log_dir,
            log_frequency=10,
            use_tensorboard=True,
            verbose=True
        ),
        CheckpointCallback(
            checkpoint_dir=args.checkpoint_dir,
            save_frequency=200,
            keep_last=3
        ),
        EvaluationCallback(
            eval_frequency=100,
            n_eval_episodes=10,
            deterministic=False,  # Use stochastic for eval
            verbose=True
        )
    ]
    
    trainer = Trainer(
        env=env,
        agent=agent,
        config=training_config,
        callbacks=callbacks
    )
    
    print(f"\nStarting training for {args.episodes} episodes...")
    print("-" * 60)
    
    stats = trainer.train()
    
    print("-" * 60)
    print("Training complete!")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Total updates: {stats['total_updates']}")
    
    env.close()


if __name__ == "__main__":
    main()
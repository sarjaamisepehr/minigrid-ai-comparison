#!/usr/bin/env python
"""Main training script for single agent."""
import argparse
from pathlib import Path
import torch
import yaml

from src.environments import MiniGridEnvironment
from src.agents import ActorCriticAgent, ActiveInferenceAgent
from src.training import Trainer, LoggingCallback, CheckpointCallback, EvaluationCallback
from src.utils import load_config, merge_configs, set_seed


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
        default="MiniGrid-Empty-8x8-v0",
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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (overrides defaults)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
        
    # Environment config
    env_config = {
        "env_id": args.env,
        "observation_type": "flat",
        "fully_observable": False,
        "max_steps": 100,
        "render_mode": None
    }
    
    # Agent config
    if args.agent == "actor_critic":
        agent_config = config.get("agent", {
            "network": {"hidden_dims": [128, 128], "activation": "relu"},
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_steps": 5
        })
    else:  # active_inference
        agent_config = config.get("agent", {
            "world_model": {
                "deterministic_dim": 128,
                "stochastic_dim": 32,
                "hidden_dim": 128,
                "embedding_dim": 128
            },
            "planning": {
                "horizon": 5,
                "num_samples": 30,
                "temperature": 1.0,
                "epistemic_weight": 1.0
            },
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "beta_kl": 1.0
        })
        
    # Training config
    training_config = config.get("training", {
        "total_episodes": args.episodes,
        "max_steps_per_episode": 100,
        "batch_size": 64,
        "buffer_capacity": 100000,
        "sequence_length": 50,
        "updates_per_episode": 4,
        "warmup_episodes": 10,
        "log_dir": args.log_dir,
        "agent_name": args.agent
    })
    
    # Create environment
    print(f"Creating environment: {args.env}")
    env = MiniGridEnvironment(env_config)
    
    # Create agent
    print(f"Creating agent: {args.agent}")
    obs_shape = env.get_observation_shape()
    action_dim = env.get_action_dim()
    
    if args.agent == "actor_critic":
        agent = ActorCriticAgent(
            observation_shape=obs_shape,
            action_dim=action_dim,
            config=agent_config,
            device=args.device
        )
    else:
        agent = ActiveInferenceAgent(
            observation_shape=obs_shape,
            action_dim=action_dim,
            config=agent_config,
            device=args.device
        )
        
    # Create callbacks
    callbacks = [
        LoggingCallback(
            log_dir=args.log_dir,
            log_frequency=10,
            use_tensorboard=True,
            verbose=True
        ),
        CheckpointCallback(
            checkpoint_dir=args.checkpoint_dir,
            save_frequency=100,
            keep_last=5
        ),
        EvaluationCallback(
            eval_frequency=100,
            n_eval_episodes=10,
            deterministic=True,
            verbose=True
        )
    ]
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        config=training_config,
        callbacks=callbacks
    )
    
    # Train
    print(f"\nStarting training for {args.episodes} episodes...")
    print(f"Device: {args.device}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")
    print("-" * 50)
    
    stats = trainer.train()
    
    print("-" * 50)
    print("Training complete!")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Total updates: {stats['total_updates']}")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
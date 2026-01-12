#!/usr/bin/env python
"""Evaluation script for trained agents."""
import argparse
from pathlib import Path
import torch

from src.environments import MiniGridEnvironment
from src.agents import ActorCriticAgent, ActiveInferenceAgent
from src.evaluation import Evaluator
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to agent checkpoint"
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["actor_critic", "active_inference"],
        required=True,
        help="Agent type"
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
        default=100,
        help="Number of evaluation episodes"
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
        default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    # Environment config
    env_config = {
        "env_id": args.env,
        "observation_type": "flat",
        "fully_observable": False,
        "max_steps": 100,
        "render_mode": "human" if args.render else None
    }
    
    # Create environment
    print(f"Creating environment: {args.env}")
    env = MiniGridEnvironment(env_config)
    
    # Load checkpoint to get config
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    agent_config = checkpoint.get("config", {})
    
    # Create agent
    obs_shape = env.get_observation_shape()
    action_dim = env.get_action_dim()
    
    print(f"Loading {args.agent} agent from {args.checkpoint}")
    
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
        
    agent.load(args.checkpoint)
    
    # Create evaluator
    evaluator = Evaluator(
        env=env,
        n_episodes=args.episodes,
        deterministic=True,
        seed=args.seed
    )
    
    # Run evaluation
    print(f"\nEvaluating for {args.episodes} episodes...")
    results = evaluator.evaluate(agent, verbose=True)
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    
    metrics = results["metrics"]
    print(f"Mean Reward:  {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Median Reward: {metrics['median_reward']:.2f}")
    print(f"Min/Max Reward: {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")
    print(f"Mean Length:  {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    
    if "success_rate" in metrics:
        print(f"Success Rate: {metrics['success_rate']:.1%}")
        
    if "reward_ci_lower" in metrics:
        print(f"95% CI: [{metrics['reward_ci_lower']:.2f}, {metrics['reward_ci_upper']:.2f}]")
        
    # Save results
    evaluator.save_results(results, args.output)
    print(f"\nResults saved to {args.output}")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
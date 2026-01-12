#!/usr/bin/env python
"""Script for comparing multiple agents."""
import argparse
from pathlib import Path
import json
import torch

from src.environments import MiniGridEnvironment
from src.agents import ActorCriticAgent, ActiveInferenceAgent
from src.training import Trainer, LoggingCallback, CheckpointCallback
from src.evaluation import Evaluator, plot_learning_curves, plot_comparison
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compare agents")
    
    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-Empty-8x8-v0",
        help="MiniGrid environment ID"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Training episodes per agent"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Evaluation episodes"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for multiple runs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Output directory"
    )
    
    return parser.parse_args()


def train_agent(
    agent_type: str,
    env_config: dict,
    agent_config: dict,
    training_config: dict,
    device: str,
    seed: int
) -> tuple:
    """Train single agent and return results."""
    set_seed(seed)
    
    env = MiniGridEnvironment(env_config)
    obs_shape = env.get_observation_shape()
    action_dim = env.get_action_dim()
    
    if agent_type == "actor_critic":
        agent = ActorCriticAgent(
            observation_shape=obs_shape,
            action_dim=action_dim,
            config=agent_config,
            device=device
        )
    else:
        agent = ActiveInferenceAgent(
            observation_shape=obs_shape,
            action_dim=action_dim,
            config=agent_config,
            device=device
        )
        
    # Simple logging callback to collect rewards
    rewards = []
    
    class RewardCollector:
        def __init__(self):
            self.rewards = []
            
        def on_training_start(self, trainer): pass
        def on_training_end(self, trainer): pass
        def on_episode_start(self, trainer, episode): pass
        def on_step(self, trainer, step, metrics): pass
        def on_update(self, trainer, update, metrics): pass
        
        def on_episode_end(self, trainer, episode, reward, length, metrics):
            self.rewards.append(reward)
            
    collector = RewardCollector()
    
    trainer = Trainer(
        env=env,
        agent=agent,
        config=training_config,
        callbacks=[collector]
    )
    
    trainer.train()
    env.close()
    
    return agent, collector.rewards


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment config
    env_config = {
        "env_id": args.env,
        "observation_type": "flat",
        "fully_observable": False,
        "max_steps": 100,
        "render_mode": None
    }
    
    # Agent configs
    ac_config = {
        "network": {"hidden_dims": [128, 128], "activation": "relu"},
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_steps": 5
    }
    
    ai_config = {
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
    }
    
    # Training config
    training_config = {
        "total_episodes": args.episodes,
        "max_steps_per_episode": 100,
        "batch_size": 64,
        "buffer_capacity": 50000,
        "sequence_length": 50,
        "updates_per_episode": 4,
        "warmup_episodes": 10
    }
    
    # Storage for results
    all_results = {
        "actor_critic": {"rewards": [], "eval_rewards": []},
        "active_inference": {"rewards": [], "eval_rewards": []}
    }
    
    # Train and evaluate for each seed
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Running with seed {seed}")
        print(f"{'='*60}")
        
        # Actor-Critic
        print(f"\nTraining Actor-Critic (seed={seed})...")
        training_config["agent_name"] = f"actor_critic_s{seed}"
        ac_agent, ac_rewards = train_agent(
            "actor_critic", env_config, ac_config, training_config, args.device, seed
        )
        all_results["actor_critic"]["rewards"].append(ac_rewards)
        
        # Active Inference
        print(f"\nTraining Active Inference (seed={seed})...")
        training_config["agent_name"] = f"active_inference_s{seed}"
        ai_agent, ai_rewards = train_agent(
            "active_inference", env_config, ai_config, training_config, args.device, seed
        )
        all_results["active_inference"]["rewards"].append(ai_rewards)
        
        # Evaluate both agents
        print(f"\nEvaluating agents (seed={seed})...")
        env = MiniGridEnvironment(env_config)
        evaluator = Evaluator(env, n_episodes=args.eval_episodes, seed=seed)
        
        ac_eval = evaluator.evaluate(ac_agent)
        ai_eval = evaluator.evaluate(ai_agent)
        
        all_results["actor_critic"]["eval_rewards"].extend(ac_eval["rewards"])
        all_results["active_inference"]["eval_rewards"].extend(ai_eval["rewards"])
        
        print(f"  Actor-Critic: {ac_eval['metrics']['mean_reward']:.2f} ± {ac_eval['metrics']['std_reward']:.2f}")
        print(f"  Active Inference: {ai_eval['metrics']['mean_reward']:.2f} ± {ai_eval['metrics']['std_reward']:.2f}")
        
        env.close()
        
    # Plot learning curves
    print("\nGenerating plots...")
    
    # Average learning curves across seeds
    avg_rewards = {}
    for agent_name, data in all_results.items():
        rewards_list = data["rewards"]
        min_len = min(len(r) for r in rewards_list)
        truncated = [r[:min_len] for r in rewards_list]
        avg_rewards[agent_name] = list(np.mean(truncated, axis=0))
        
    import numpy as np
    
    fig = plot_learning_curves(
        avg_rewards,
        title=f"Learning Curves - {args.env}",
        smooth_window=50,
        save_path=output_dir / "learning_curves.png"
    )
    
    # Final comparison
    print("\n" + "=" * 60)
    print("Final Comparison Results")
    print("=" * 60)
    
    from src.evaluation.metrics import statistical_comparison
    
    comparison = statistical_comparison(
        all_results["actor_critic"]["eval_rewards"],
        all_results["active_inference"]["eval_rewards"]
    )
    
    print(f"\nActor-Critic:      {comparison['mean_a']:.2f} ± {comparison['std_a']:.2f}")
    print(f"Active Inference:  {comparison['mean_b']:.2f} ± {comparison['std_b']:.2f}")
    print(f"\nEffect Size (Cohen's d): {comparison['effect_size']:.3f}")
    print(f"p-value: {comparison['p_value']:.4f}")
    
    if comparison['p_value'] < 0.05:
        if comparison['mean_a'] > comparison['mean_b']:
            print("\n→ Actor-Critic significantly outperforms Active Inference")
        else:
            print("\n→ Active Inference significantly outperforms Actor-Critic")
    else:
        print("\n→ No significant difference between agents")
        
    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "config": {
                "env": args.env,
                "episodes": args.episodes,
                "eval_episodes": args.eval_episodes,
                "seeds": args.seeds
            },
            "comparison": comparison,
            "rewards": {
                k: {
                    "training": v["rewards"],
                    "evaluation": v["eval_rewards"]
                }
                for k, v in all_results.items()
            }
        }, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
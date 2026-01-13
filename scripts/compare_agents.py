#!/usr/bin/env python
"""Script for comparing Actor-Critic vs Active Inference agents."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
import torch

from src.environments import MiniGridEnvironment
from src.agents import ActorCriticAgent, ActiveInferenceAgent
from src.training import Trainer, LoggingCallback
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compare AC vs AI agents")
    
    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-Empty-5x5-v0",
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
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Output directory"
    )
    
    return parser.parse_args()


class MetricsCollector:
    """Simple callback to collect training metrics."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.wm_losses = []
        
    def on_training_start(self, trainer): 
        self.episode_rewards = []
        self.episode_lengths = []
        self.wm_losses = []
        
    def on_training_end(self, trainer): pass
    def on_episode_start(self, trainer, episode): pass
    def on_step(self, trainer, step, metrics): pass
    def on_update(self, trainer, update, metrics): pass
    
    def on_episode_end(self, trainer, episode, reward, length, metrics):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if "world_model_loss" in metrics:
            wm_loss = metrics["world_model_loss"]
            if isinstance(wm_loss, list):
                wm_loss = np.mean(wm_loss)
            self.wm_losses.append(wm_loss)


def evaluate_agent(env, agent, n_episodes=50, max_steps=100):
    """Evaluate agent performance."""
    agent.eval_mode()
    
    rewards = []
    lengths = []
    successes = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(agent, 'reset_belief'):
            agent.reset_belief()
            
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            output = agent.act(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(output.action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
        rewards.append(episode_reward)
        lengths.append(episode_length)
        successes.append(float(episode_reward > 0))
        
    agent.train_mode()
    
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "success_rate": float(np.mean(successes)),
        "rewards": rewards,
        "lengths": lengths
    }


def train_actor_critic(env, device, episodes, seed):
    """Train Actor-Critic agent."""
    set_seed(seed)
    
    obs_shape = env.get_observation_shape()
    action_dim = env.get_action_dim()
    
    agent_config = {
        "network": {"hidden_dims": [64, 64], "activation": "relu"},
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
        device=device
    )
    
    training_config = {
        "total_episodes": episodes,
        "max_steps_per_episode": 100,
        "batch_size": 8,
        "buffer_capacity": 5000,
        "warmup_episodes": 3,
        "agent_name": "actor_critic",
        "log_dir": "logs"
    }
    
    collector = MetricsCollector()
    
    trainer = Trainer(
        env=env,
        agent=agent,
        config=training_config,
        callbacks=[collector]
    )
    
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    return agent, collector, train_time


def train_active_inference(env, device, episodes, seed):
    """Train Active Inference agent."""
    set_seed(seed)
    
    obs_shape = env.get_observation_shape()
    action_dim = env.get_action_dim()
    
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
        device=device
    )
    
    training_config = {
        "total_episodes": episodes,
        "max_steps_per_episode": 100,
        "batch_size": 8,
        "buffer_capacity": 5000,
        "sequence_length": 15,
        "updates_per_episode": 2,
        "warmup_episodes": 3,
        "agent_name": "active_inference",
        "log_dir": "logs"
    }
    
    collector = MetricsCollector()
    
    trainer = Trainer(
        env=env,
        agent=agent,
        config=training_config,
        callbacks=[collector]
    )
    
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    return agent, collector, train_time


def compute_sample_efficiency(rewards, threshold=0.5, window=50):
    """Compute episodes needed to reach threshold performance."""
    if len(rewards) < window:
        return len(rewards)
    
    running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    above_threshold = np.where(running_avg >= threshold)[0]
    
    if len(above_threshold) > 0:
        return above_threshold[0] + window
    return len(rewards)  # Never reached threshold


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AGENT COMPARISON: Actor-Critic vs Active Inference")
    print("=" * 70)
    print(f"Environment: {args.env}")
    print(f"Training episodes: {args.episodes}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # Environment config
    env_config = {
        "env_id": args.env,
        "observation_type": "flat",
        "fully_observable": False,
        "max_steps": 100,
        "render_mode": None
    }
    
    # ==================== ACTOR-CRITIC ====================
    print("\n" + "=" * 70)
    print("TRAINING ACTOR-CRITIC")
    print("=" * 70)
    
    env = MiniGridEnvironment(env_config)
    ac_agent, ac_collector, ac_train_time = train_actor_critic(
        env, args.device, args.episodes, args.seed
    )
    
    print(f"\nEvaluating Actor-Critic ({args.eval_episodes} episodes)...")
    ac_eval = evaluate_agent(env, ac_agent, args.eval_episodes)
    env.close()
    
    print(f"  Mean Reward: {ac_eval['mean_reward']:.3f} ± {ac_eval['std_reward']:.3f}")
    print(f"  Success Rate: {ac_eval['success_rate']:.1%}")
    print(f"  Mean Length: {ac_eval['mean_length']:.1f}")
    print(f"  Training Time: {ac_train_time:.1f}s")
    
    # ==================== ACTIVE INFERENCE ====================
    print("\n" + "=" * 70)
    print("TRAINING ACTIVE INFERENCE")
    print("=" * 70)
    
    env = MiniGridEnvironment(env_config)
    ai_agent, ai_collector, ai_train_time = train_active_inference(
        env, args.device, args.episodes, args.seed
    )
    
    print(f"\nEvaluating Active Inference ({args.eval_episodes} episodes)...")
    ai_eval = evaluate_agent(env, ai_agent, args.eval_episodes)
    env.close()
    
    print(f"  Mean Reward: {ai_eval['mean_reward']:.3f} ± {ai_eval['std_reward']:.3f}")
    print(f"  Success Rate: {ai_eval['success_rate']:.1%}")
    print(f"  Mean Length: {ai_eval['mean_length']:.1f}")
    print(f"  Training Time: {ai_train_time:.1f}s")
    
    # ==================== COMPARISON ====================
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    # Sample efficiency
    ac_efficiency = compute_sample_efficiency(ac_collector.episode_rewards)
    ai_efficiency = compute_sample_efficiency(ai_collector.episode_rewards)
    
    # Learning curves statistics
    ac_final_100 = np.mean(ac_collector.episode_rewards[-100:]) if len(ac_collector.episode_rewards) >= 100 else np.mean(ac_collector.episode_rewards)
    ai_final_100 = np.mean(ai_collector.episode_rewards[-100:]) if len(ai_collector.episode_rewards) >= 100 else np.mean(ai_collector.episode_rewards)
    
    # Print comparison table
    print("\n{:<30} {:>15} {:>15}".format("Metric", "Actor-Critic", "Active Inf."))
    print("-" * 62)
    print("{:<30} {:>15.3f} {:>15.3f}".format("Final Eval Reward", ac_eval['mean_reward'], ai_eval['mean_reward']))
    print("{:<30} {:>15.1%} {:>15.1%}".format("Success Rate", ac_eval['success_rate'], ai_eval['success_rate']))
    print("{:<30} {:>15.1f} {:>15.1f}".format("Mean Episode Length", ac_eval['mean_length'], ai_eval['mean_length']))
    print("{:<30} {:>15.3f} {:>15.3f}".format("Final Training Reward (100)", ac_final_100, ai_final_100))
    print("{:<30} {:>15d} {:>15d}".format("Episodes to 50% Success", ac_efficiency, ai_efficiency))
    print("{:<30} {:>15.1f}s {:>14.1f}s".format("Training Time", ac_train_time, ai_train_time))
    print("{:<30} {:>15.1f} {:>15.1f}".format("Episodes/Second", args.episodes/ac_train_time, args.episodes/ai_train_time))
    
    # Winner determination
    print("\n" + "-" * 62)
    
    if ac_eval['mean_reward'] > ai_eval['mean_reward'] + 0.05:
        print("WINNER (Performance): Actor-Critic")
    elif ai_eval['mean_reward'] > ac_eval['mean_reward'] + 0.05:
        print("WINNER (Performance): Active Inference")
    else:
        print("RESULT (Performance): Roughly Equal")
        
    if ac_efficiency < ai_efficiency:
        print("WINNER (Sample Efficiency): Actor-Critic")
    elif ai_efficiency < ac_efficiency:
        print("WINNER (Sample Efficiency): Active Inference")
    else:
        print("RESULT (Sample Efficiency): Roughly Equal")
        
    if ac_train_time < ai_train_time:
        print("WINNER (Training Speed): Actor-Critic")
    else:
        print("WINNER (Training Speed): Active Inference")
    
    # ==================== SAVE RESULTS ====================
    results = {
        "config": {
            "environment": args.env,
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed
        },
        "actor_critic": {
            "eval": ac_eval,
            "training_rewards": ac_collector.episode_rewards,
            "training_lengths": ac_collector.episode_lengths,
            "train_time": ac_train_time,
            "sample_efficiency": ac_efficiency,
            "final_100_reward": float(ac_final_100)
        },
        "active_inference": {
            "eval": ai_eval,
            "training_rewards": ai_collector.episode_rewards,
            "training_lengths": ai_collector.episode_lengths,
            "wm_losses": ai_collector.wm_losses,
            "train_time": ai_train_time,
            "sample_efficiency": ai_efficiency,
            "final_100_reward": float(ai_final_100)
        }
    }
    
    results_path = output_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nResults saved to: {results_path}")
    
    # ==================== PLOT ====================
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Learning curves
        ax1 = axes[0, 0]
        window = 50
        
        if len(ac_collector.episode_rewards) >= window:
            ac_smooth = np.convolve(ac_collector.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(ac_collector.episode_rewards)), ac_smooth, label='Actor-Critic', color='blue')
        ax1.plot(ac_collector.episode_rewards, alpha=0.2, color='blue')
        
        if len(ai_collector.episode_rewards) >= window:
            ai_smooth = np.convolve(ai_collector.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(ai_collector.episode_rewards)), ai_smooth, label='Active Inference', color='orange')
        ax1.plot(ai_collector.episode_rewards, alpha=0.2, color='orange')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode lengths
        ax2 = axes[0, 1]
        if len(ac_collector.episode_lengths) >= window:
            ac_len_smooth = np.convolve(ac_collector.episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(ac_collector.episode_lengths)), ac_len_smooth, label='Actor-Critic', color='blue')
        if len(ai_collector.episode_lengths) >= window:
            ai_len_smooth = np.convolve(ai_collector.episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(ai_collector.episode_lengths)), ai_len_smooth, label='Active Inference', color='orange')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths (lower is better)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Final performance comparison
        ax3 = axes[1, 0]
        metrics = ['Mean Reward', 'Success Rate', 'Efficiency\n(normalized)']
        ac_values = [ac_eval['mean_reward'], ac_eval['success_rate'], 1 - ac_efficiency/args.episodes]
        ai_values = [ai_eval['mean_reward'], ai_eval['success_rate'], 1 - ai_efficiency/args.episodes]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, ac_values, width, label='Actor-Critic', color='blue')
        ax3.bar(x + width/2, ai_values, width, label='Active Inference', color='orange')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.set_ylabel('Value')
        ax3.set_title('Final Performance Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # World model loss (Active Inference only)
        ax4 = axes[1, 1]
        if ai_collector.wm_losses:
            ax4.plot(ai_collector.wm_losses, color='orange')
            ax4.set_xlabel('Update')
            ax4.set_ylabel('Loss')
            ax4.set_title('Active Inference: World Model Loss')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No world model loss data', ha='center', va='center')
            ax4.set_title('Active Inference: World Model Loss')
        
        plt.tight_layout()
        plot_path = output_dir / "comparison_plot.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    except Exception as e:
        print(f"\nPlotting failed: {e}")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
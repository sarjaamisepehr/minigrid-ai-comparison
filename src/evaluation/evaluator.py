"""Evaluation runner for agent comparison."""
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np

from ..environments.base_env import BaseEnvironment
from ..agents.base_agent import BaseAgent
from .metrics import compute_metrics, statistical_comparison


class Evaluator:
    """
    Evaluator for running standardized agent evaluations.
    
    Supports:
    - Single agent evaluation
    - Multi-agent comparison
    - Statistical significance testing
    """
    
    def __init__(
        self,
        env: BaseEnvironment,
        n_episodes: int = 100,
        deterministic: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
            seed: Random seed for reproducibility
        """
        self.env = env
        self.n_episodes = n_episodes
        self.deterministic = deterministic
        self.seed = seed
        
    def evaluate(
        self,
        agent: BaseAgent,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate single agent.
        
        Args:
            agent: Agent to evaluate
            verbose: Print progress
            
        Returns:
            Evaluation results
        """
        agent.eval_mode()
        
        rewards = []
        lengths = []
        successes = []
        
        for ep in range(self.n_episodes):
            seed = self.seed + ep if self.seed is not None else None
            episode_data = self._run_episode(agent, seed)
            
            rewards.append(episode_data["reward"])
            lengths.append(episode_data["length"])
            successes.append(episode_data["success"])
            
            if verbose and (ep + 1) % 10 == 0:
                print(f"Evaluated {ep + 1}/{self.n_episodes} episodes")
                
        agent.train_mode()
        
        metrics = compute_metrics(rewards, lengths, successes)
        
        return {
            "metrics": metrics,
            "rewards": rewards,
            "lengths": lengths,
            "successes": successes
        }
    
    def _run_episode(
        self,
        agent: BaseAgent,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run single evaluation episode."""
        obs, info = self.env.reset(seed=seed)
        
        if hasattr(agent, 'reset_belief'):
            agent.reset_belief()
            
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            output = agent.act(obs, deterministic=self.deterministic)
            obs, reward, terminated, truncated, info = self.env.step(output.action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
        return {
            "reward": episode_reward,
            "length": episode_length,
            "success": episode_reward > 0
        }
    
    def compare(
        self,
        agents: Dict[str, BaseAgent],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple agents.
        
        Args:
            agents: Dictionary mapping agent names to agents
            verbose: Print results
            
        Returns:
            Comparison results with statistical tests
        """
        results = {}
        
        # Evaluate each agent
        for name, agent in agents.items():
            if verbose:
                print(f"\nEvaluating {name}...")
                
            results[name] = self.evaluate(agent, verbose=verbose)
            
            if verbose:
                metrics = results[name]["metrics"]
                print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                print(f"  Success Rate: {metrics.get('success_rate', 0):.1%}")
                
        # Statistical comparisons
        agent_names = list(agents.keys())
        comparisons = {}
        
        for i, name_a in enumerate(agent_names):
            for name_b in agent_names[i + 1:]:
                comparison_key = f"{name_a}_vs_{name_b}"
                comparisons[comparison_key] = statistical_comparison(
                    results[name_a]["rewards"],
                    results[name_b]["rewards"],
                    test="welch"
                )
                
                if verbose:
                    comp = comparisons[comparison_key]
                    print(f"\n{name_a} vs {name_b}:")
                    print(f"  Effect Size (Cohen's d): {comp['effect_size']:.3f}")
                    print(f"  p-value: {comp['p_value']:.4f}")
                    
        return {
            "individual_results": results,
            "comparisons": comparisons
        }
    
    def save_results(
        self,
        results: Dict[str, Any],
        path: str
    ) -> None:
        """Save evaluation results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
            
        with open(path, 'w') as f:
            json.dump(convert(results), f, indent=2)
"""Evaluation metrics for agent comparison."""
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


def compute_metrics(
    rewards: List[float],
    lengths: List[int],
    successes: Optional[List[bool]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics from episode data.
    
    Args:
        rewards: List of episode rewards
        lengths: List of episode lengths
        successes: Optional list of success indicators
        
    Returns:
        Dictionary of computed metrics
    """
    rewards = np.array(rewards)
    lengths = np.array(lengths)
    
    metrics = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "median_reward": np.median(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }
    
    # Confidence interval for mean reward
    if len(rewards) > 1:
        ci = stats.t.interval(
            0.95,
            len(rewards) - 1,
            loc=np.mean(rewards),
            scale=stats.sem(rewards)
        )
        metrics["reward_ci_lower"] = ci[0]
        metrics["reward_ci_upper"] = ci[1]
    
    if successes is not None:
        successes = np.array(successes, dtype=float)
        metrics["success_rate"] = np.mean(successes)
        
        # Wilson score interval for success rate
        n = len(successes)
        p = np.mean(successes)
        z = 1.96  # 95% confidence
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        metrics["success_ci_lower"] = max(0, center - spread)
        metrics["success_ci_upper"] = min(1, center + spread)
        
    return metrics


def compute_sample_efficiency(
    rewards_over_time: List[List[float]],
    threshold: float,
    window_size: int = 100
) -> Dict[str, float]:
    """
    Compute sample efficiency metrics.
    
    Args:
        rewards_over_time: List of reward histories per seed
        threshold: Performance threshold to reach
        window_size: Window for computing running average
        
    Returns:
        Dictionary with sample efficiency metrics
    """
    episodes_to_threshold = []
    final_performances = []
    
    for rewards in rewards_over_time:
        rewards = np.array(rewards)
        
        # Compute running average
        if len(rewards) >= window_size:
            running_avg = np.convolve(
                rewards, 
                np.ones(window_size) / window_size, 
                mode='valid'
            )
        else:
            running_avg = rewards
            
        # Find first episode where threshold is reached
        threshold_reached = np.where(running_avg >= threshold)[0]
        
        if len(threshold_reached) > 0:
            episodes_to_threshold.append(threshold_reached[0] + window_size)
        else:
            episodes_to_threshold.append(np.inf)
            
        # Final performance (last window average)
        final_performances.append(running_avg[-1] if len(running_avg) > 0 else 0)
        
    episodes_to_threshold = np.array(episodes_to_threshold)
    final_performances = np.array(final_performances)
    
    # Filter out inf values for statistics
    finite_episodes = episodes_to_threshold[np.isfinite(episodes_to_threshold)]
    
    metrics = {
        "mean_episodes_to_threshold": np.mean(finite_episodes) if len(finite_episodes) > 0 else np.inf,
        "std_episodes_to_threshold": np.std(finite_episodes) if len(finite_episodes) > 1 else 0,
        "success_rate_reaching_threshold": np.mean(np.isfinite(episodes_to_threshold)),
        "mean_final_performance": np.mean(final_performances),
        "std_final_performance": np.std(final_performances)
    }
    
    return metrics


def compute_area_under_curve(
    rewards: List[float],
    normalize: bool = True
) -> float:
    """
    Compute area under the learning curve.
    
    Higher AUC indicates better overall learning efficiency.
    
    Args:
        rewards: Episode rewards over training
        normalize: Whether to normalize by number of episodes
        
    Returns:
        Area under curve value
    """
    rewards = np.array(rewards)
    auc = np.trapz(rewards)
    
    if normalize:
        auc /= len(rewards)
        
    return auc


def statistical_comparison(
    rewards_a: List[float],
    rewards_b: List[float],
    test: str = "welch"
) -> Dict[str, float]:
    """
    Statistical comparison between two agents.
    
    Args:
        rewards_a: Rewards from agent A
        rewards_b: Rewards from agent B
        test: Statistical test ("welch", "mannwhitney", "bootstrap")
        
    Returns:
        Dictionary with test statistics and p-value
    """
    rewards_a = np.array(rewards_a)
    rewards_b = np.array(rewards_b)
    
    results = {
        "mean_a": np.mean(rewards_a),
        "mean_b": np.mean(rewards_b),
        "std_a": np.std(rewards_a),
        "std_b": np.std(rewards_b),
        "effect_size": (np.mean(rewards_a) - np.mean(rewards_b)) / np.sqrt(
            (np.std(rewards_a)**2 + np.std(rewards_b)**2) / 2
        )  # Cohen's d
    }
    
    if test == "welch":
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(rewards_a, rewards_b, equal_var=False)
        results["t_statistic"] = t_stat
        results["p_value"] = p_value
        
    elif test == "mannwhitney":
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(
            rewards_a, rewards_b, alternative='two-sided'
        )
        results["u_statistic"] = u_stat
        results["p_value"] = p_value
        
    elif test == "bootstrap":
        # Bootstrap confidence interval for difference
        n_bootstrap = 10000
        diffs = []
        
        for _ in range(n_bootstrap):
            sample_a = np.random.choice(rewards_a, size=len(rewards_a), replace=True)
            sample_b = np.random.choice(rewards_b, size=len(rewards_b), replace=True)
            diffs.append(np.mean(sample_a) - np.mean(sample_b))
            
        diffs = np.array(diffs)
        results["bootstrap_mean_diff"] = np.mean(diffs)
        results["bootstrap_ci_lower"] = np.percentile(diffs, 2.5)
        results["bootstrap_ci_upper"] = np.percentile(diffs, 97.5)
        # P-value approximation
        results["p_value"] = 2 * min(
            np.mean(diffs <= 0),
            np.mean(diffs >= 0)
        )
        
    return results
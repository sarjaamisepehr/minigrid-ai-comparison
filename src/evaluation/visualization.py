"""Visualization utilities for training results."""
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def smooth_curve(
    values: List[float],
    window: int = 100
) -> np.ndarray:
    """Smooth curve using moving average."""
    values = np.array(values)
    if len(values) < window:
        return values
        
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='valid')
    return smoothed


def plot_learning_curves(
    results: Dict[str, List[float]],
    title: str = "Learning Curves",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    smooth_window: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot learning curves for multiple agents.
    
    Args:
        results: Dictionary mapping agent names to reward lists
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        smooth_window: Smoothing window size
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (name, rewards), color in zip(results.items(), colors):
        rewards = np.array(rewards)
        episodes = np.arange(len(rewards))
        
        # Plot raw data with transparency
        ax.plot(episodes, rewards, alpha=0.2, color=color)
        
        # Plot smoothed curve
        if len(rewards) >= smooth_window:
            smoothed = smooth_curve(rewards, smooth_window)
            smooth_episodes = np.arange(
                smooth_window - 1,
                smooth_window - 1 + len(smoothed)
            )
            ax.plot(
                smooth_episodes,
                smoothed,
                label=name,
                color=color,
                linewidth=2
            )
        else:
            ax.plot(episodes, rewards, label=name, color=color, linewidth=2)
            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = "mean_reward",
    title: str = "Agent Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot bar comparison of agents.
    
    Args:
        results: Evaluation results from Evaluator.compare()
        metric: Metric to compare
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(results.keys())
    means = [results[name]["metrics"][metric] for name in names]
    stds = [results[name]["metrics"].get(f"std_{metric.split('_')[-1]}", 0) for name in names]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors)
    
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + std + 0.01 * max(means),
            f'{mean:.2f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_sample_efficiency(
    results: Dict[str, Dict[str, List[float]]],
    threshold: float,
    title: str = "Sample Efficiency",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot sample efficiency comparison.
    
    Shows episodes required to reach performance threshold.
    
    Args:
        results: Dictionary with rewards per seed per agent
        threshold: Performance threshold
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Learning curves with threshold line
    ax1 = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (name, data), color in zip(results.items(), colors):
        all_rewards = data["rewards"]  # List of lists (per seed)
        
        # Compute mean and std across seeds
        max_len = max(len(r) for r in all_rewards)
        padded = np.array([
            np.pad(r, (0, max_len - len(r)), constant_values=np.nan)
            for r in all_rewards
        ])
        
        mean_rewards = np.nanmean(padded, axis=0)
        std_rewards = np.nanstd(padded, axis=0)
        
        episodes = np.arange(len(mean_rewards))
        
        ax1.plot(episodes, mean_rewards, label=name, color=color, linewidth=2)
        ax1.fill_between(
            episodes,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
            color=color
        )
        
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Learning Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Box plot of episodes to threshold
    ax2 = axes[1]
    
    episodes_data = []
    names = []
    
    for name, data in results.items():
        episodes_to_threshold = data.get("episodes_to_threshold", [])
        episodes_data.append([e for e in episodes_to_threshold if np.isfinite(e)])
        names.append(name)
        
    if any(len(d) > 0 for d in episodes_data):
        bp = ax2.boxplot(episodes_data, labels=names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
    ax2.set_ylabel("Episodes to Threshold")
    ax2.set_title("Sample Efficiency")
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_world_model_quality(
    losses: Dict[str, List[float]],
    title: str = "World Model Training",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot world model training losses.
    
    Args:
        losses: Dictionary with loss names and histories
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_plots = len(losses)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
        
    for ax, (name, values) in zip(axes, losses.items()):
        values = np.array(values)
        
        ax.plot(values, alpha=0.3)
        
        if len(values) >= 100:
            smoothed = smooth_curve(values, 100)
            ax.plot(
                np.arange(99, 99 + len(smoothed)),
                smoothed,
                linewidth=2
            )
            
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig
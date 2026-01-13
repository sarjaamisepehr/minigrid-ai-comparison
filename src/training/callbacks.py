"""Training callbacks for logging, checkpointing, and evaluation."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import json
import time
import numpy as np

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback(ABC):
    """Abstract base class for training callbacks."""
    
    def on_training_start(self, trainer: 'Trainer') -> None:
        """Called at the start of training."""
        pass
    
    def on_training_end(self, trainer: 'Trainer') -> None:
        """Called at the end of training."""
        pass
    
    def on_episode_start(self, trainer: 'Trainer', episode: int) -> None:
        """Called at the start of each episode."""
        pass
    
    def on_episode_end(
        self, 
        trainer: 'Trainer', 
        episode: int,
        episode_reward: float,
        episode_length: int,
        metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each episode."""
        pass
    
    def on_step(
        self,
        trainer: 'Trainer',
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """Called after each environment step."""
        pass
    
    def on_update(
        self,
        trainer: 'Trainer',
        update: int,
        metrics: Dict[str, float]
    ) -> None:
        """Called after each agent update."""
        pass


class LoggingCallback(Callback):
    """
    Callback for logging training metrics.
    
    Supports console logging and TensorBoard.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_frequency: int = 10,
        use_tensorboard: bool = True,
        verbose: bool = True
    ):
        """
        Initialize logging callback.
        
        Args:
            log_dir: Directory for log files
            log_frequency: Log every N episodes
            use_tensorboard: Whether to use TensorBoard
            verbose: Whether to print to console
        """
        self.log_dir = Path(log_dir)
        self.log_frequency = log_frequency
        self.use_tensorboard = use_tensorboard
        self.verbose = verbose
        
        self.writer = None
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.start_time: Optional[float] = None
        
    def on_training_start(self, trainer: 'Trainer') -> None:
        """Setup logging."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        
        # Reset lists
        self.episode_rewards = []
        self.episode_lengths = []
        
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                run_name = f"{trainer.agent_name}_{int(time.time())}"
                self.writer = SummaryWriter(self.log_dir / run_name)
            except ImportError:
                print("TensorBoard not available. Falling back to console logging.")
                self.use_tensorboard = False
                
    def on_training_end(self, trainer: 'Trainer') -> None:
        """Cleanup logging."""
        if self.writer is not None:
            self.writer.close()
            
        # Save final metrics
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            final_metrics = {
                "total_episodes": len(self.episode_rewards),
                "total_time": elapsed_time,
                "mean_reward": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0,
                "mean_length": float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0
            }
            
            try:
                with open(self.log_dir / "final_metrics.json", 'w') as f:
                    json.dump(final_metrics, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save final metrics: {e}")
            
    def on_episode_end(
        self,
        trainer: 'Trainer',
        episode: int,
        episode_reward: float,
        episode_length: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log episode metrics."""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        if episode % self.log_frequency == 0:
            # Compute running statistics
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]
            
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            mean_length = np.mean(recent_lengths)
            
            elapsed = time.time() - self.start_time if self.start_time else 0
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            
            if self.verbose:
                # Base output
                output = (
                    f"Episode {episode:5d} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Mean(100): {mean_reward:7.2f} ± {std_reward:.2f} | "
                    f"Length: {episode_length:4d} | "
                    f"EPS: {eps_per_sec:.2f}"
                )
                
                # Add world model loss if available (Active Inference)
                if "world_model_loss" in metrics:
                    wm_loss = metrics["world_model_loss"]
                    if isinstance(wm_loss, list):
                        wm_loss = np.mean(wm_loss)
                    output += f" | WM Loss: {wm_loss:.4f}"
                    
                print(output)
                
            if self.writer is not None:
                self.writer.add_scalar("episode/reward", episode_reward, episode)
                self.writer.add_scalar("episode/length", episode_length, episode)
                self.writer.add_scalar("episode/mean_reward_100", mean_reward, episode)
                self.writer.add_scalar("episode/mean_length_100", mean_length, episode)
                
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"train/{key}", value, episode)
                    elif isinstance(value, list) and len(value) > 0:
                        self.writer.add_scalar(f"train/{key}", np.mean(value), episode)
                    
    def on_update(
        self,
        trainer: 'Trainer',
        update: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log update metrics to TensorBoard."""
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"update/{key}", value, update)


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_frequency: int = 100,
        keep_last: int = 5
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            save_frequency: Save every N episodes
            keep_last: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_frequency = save_frequency
        self.keep_last = keep_last
        
        self.saved_checkpoints: List[Path] = []
        
    def on_training_start(self, trainer: 'Trainer') -> None:
        """Create checkpoint directory."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints = []
        
    def on_episode_end(
        self,
        trainer: 'Trainer',
        episode: int,
        episode_reward: float,
        episode_length: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save checkpoint if needed."""
        if episode % self.save_frequency == 0 and episode > 0:
            checkpoint_path = (
                self.checkpoint_dir / 
                f"{trainer.agent_name}_ep{episode}.pt"
            )
            trainer.agent.save(checkpoint_path)
            self.saved_checkpoints.append(checkpoint_path)
            
            # Remove old checkpoints
            while len(self.saved_checkpoints) > self.keep_last:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    
    def on_training_end(self, trainer: 'Trainer') -> None:
        """Save final checkpoint."""
        final_path = self.checkpoint_dir / f"{trainer.agent_name}_final.pt"
        trainer.agent.save(final_path)


class EvaluationCallback(Callback):
    """Callback for periodic evaluation."""
    
    def __init__(
        self,
        eval_frequency: int = 100,
        n_eval_episodes: int = 10,
        deterministic: bool = False,  # Changed to False!
        verbose: bool = True
    ):
        """
        Initialize evaluation callback.
        
        Args:
            eval_frequency: Evaluate every N episodes
            n_eval_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy for evaluation
            verbose: Print evaluation results
        """
        self.eval_frequency = eval_frequency
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.verbose = verbose
        
        self.eval_results: List[Dict[str, float]] = []
        
    def on_training_start(self, trainer: 'Trainer') -> None:
        """Reset evaluation results."""
        self.eval_results = []
        
    def on_episode_end(
        self,
        trainer: 'Trainer',
        episode: int,
        episode_reward: float,
        episode_length: int,
        metrics: Dict[str, float]
    ) -> None:
        """Run evaluation if needed."""
        if episode % self.eval_frequency == 0 and episode > 0:
            eval_metrics = self._evaluate(trainer)
            self.eval_results.append({
                "episode": episode,
                **eval_metrics
            })
            
            if self.verbose:
                print(
                    f"  [Eval] Mean Reward: {eval_metrics['mean_reward']:.2f} ± "
                    f"{eval_metrics['std_reward']:.2f} | "
                    f"Success Rate: {eval_metrics['success_rate']:.1%}"
                )
                
    def _evaluate(self, trainer: 'Trainer') -> Dict[str, float]:
        """Run evaluation episodes."""
        trainer.agent.eval_mode()
        
        rewards = []
        lengths = []
        successes = []
        
        max_steps = trainer.max_steps  # Get max steps from trainer
        
        for ep in range(self.n_eval_episodes):
            obs, info = trainer.env.reset()
            
            # Reset agent state if needed
            if hasattr(trainer.agent, 'reset_belief'):
                trainer.agent.reset_belief()
                
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done and episode_length < max_steps:
                output = trainer.agent.act(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = trainer.env.step(output.action)
                
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(float(episode_reward > 0))
            
        trainer.agent.train_mode()
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes))
        }
    
    def on_training_end(self, trainer: 'Trainer') -> None:
        """Save evaluation results."""
        if self.eval_results:
            results_path = Path(trainer.log_dir) / "eval_results.json"
            try:
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, 'w') as f:
                    json.dump(self.eval_results, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save eval results: {e}")
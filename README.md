# MiniGrid AI Comparison

Comparing Active Inference agents with RSSM world models against model-free Actor-Critic agents in MiniGrid environments.

## Project Overview

This project implements a modular framework for comparing:
- **Active Inference Agent**: Uses RSSM world model with Expected Free Energy minimization
- **Actor-Critic Agent**: Model-free A2C for discrete action spaces

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/minigrid-ai-comparison.git
cd minigrid-ai-comparison

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start
```python
from src.environments import MiniGridEnvironment

# Create environment
config = {
    "env_id": "MiniGrid-Empty-8x8-v0",
    "observation_type": "flat",
    "max_steps": 100
}
env = MiniGridEnvironment(config)

# Run episode
obs, info = env.reset(seed=42)
done = False

while not done:
    action = env.sample_random_action()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

## Project Structure
```
minigrid-ai-comparison/
├── configs/          # YAML configuration files
├── src/              # Source code
│   ├── environments/ # Environment wrappers
│   ├── agents/       # Agent implementations
│   ├── training/     # Training infrastructure
│   └── evaluation/   # Evaluation metrics
├── scripts/          # Training/evaluation scripts
└── tests/            # Unit tests
```

## Running Tests
```bash
pytest tests/ -v
```

## License

MIT License"# minigrid-ai-comparison" 

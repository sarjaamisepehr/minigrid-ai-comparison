from setuptools import setup, find_packages

setup(
    name="minigrid-ai-comparison",
    version="0.1.0",
    author="Sepehr",
    description="Comparing Active Inference and RL agents in MiniGrid",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "minigrid>=2.3.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
# PyBoy Tetris CNN Reinforcement Learning

A deep reinforcement learning project that trains a convolutional neural network to play Tetris through PyBoy emulation using a hybrid approach of evolutionary algorithms and gradient-based learning.

![Tetris AI Demo](data/demo.gif)

## Project Overview

This project implements a hybrid reinforcement learning system to train agents to play Tetris on the Game Boy. The system combines:

- **Deep Q-Learning**: Using convolutional neural networks to process game screen data
- **Gradient-Based Learning**: Traditional backpropagation for efficient parameter optimization
- **Evolutionary Algorithm**: Using genetic mutations to explore parameter space across generations
- **PyBoy**: For Game Boy emulation to play Tetris in a controlled environment
- **Parallel Training**: Multi-process training to evaluate multiple agents simultaneously

The training process combines the strengths of both evolutionary and gradient-based approaches:
1. Multiple agents play the game and collect experiences
2. Each agent performs gradient-based training on its collected experiences
3. At the end of each generation, the best-performing agents are selected
4. New agents are created through mutations of the best performers
5. The cycle repeats, combining the exploration benefits of evolution with the efficiency of gradient descent

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- PyBoy 1.4.2+
- Game Boy Tetris ROM (not included)
- Other dependencies in `requirements.txt`

## Project Structure

```
conv-pyboy/
├── data/                   # For storing training data, checkpoints, and recordings
│   ├── agents/            # Agent checkpoints
│   ├── logs/              # TensorBoard logs
│   └── recordings/        # Game play recordings
├── models/                # Neural network architectures
│   └── cnn_model.py       # CNN model implementations
├── agents/                # Agent implementations
│   ├── dqn_agent.py       # DQN agent implementation
│   ├── population.py      # Population management for hybrid learning
│   └── worker.py          # Worker process for parallel training
├── environment/           # Game environment wrapper
│   └── tetris_env.py      # PyBoy Tetris environment
├── utils/                 # Helper functions
│   ├── data_utils.py      # Utilities for data processing and visualization
│   └── device_utils.py    # Utilities for GPU/CPU device management
├── configs/               # Configuration files
│   └── default_config.py  # Default hyperparameters
├── main.py                # Entry point for training
├── test_model.py          # Script for testing trained models
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/conv-pyboy.git
   cd conv-pyboy
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Obtain a Tetris ROM for Game Boy (not included due to copyright)

## Usage

### Training

To start training a new agent with the hybrid approach:

```bash
python main.py --rom_path path/to/tetris.gb --headless --population_size 10 --generations 100 --training_steps 64
```

For parallel training with multiple processes:

```bash
python main.py --rom_path path/to/tetris.gb --headless --population_size 10 --generations 100 --training_steps 64 --parallel
```

Key training parameters:
- `--training_steps`: Number of gradient updates per episode (higher values = more GPU utilization)
- `--population_size`: Number of agents in the population
- `--generations`: Number of evolutionary generations to run
- `--mutation_rate`: Probability of mutating each network parameter
- `--mutation_scale`: Scale of random mutations
- `--parallel`: Enable parallel training (multiple PyBoy instances)
- `--num_processes`: Number of parallel processes to use (defaults to min of CPU count and population size)

For a full list of options:
```bash
python main.py --help
```

### Monitor Training Progress

You can monitor the training progress with TensorBoard:

```bash
tensorboard --logdir data/logs
```

TensorBoard will show:
- Score progression across generations
- Training losses for each agent
- Network architecture visualization
- Performance metrics

### Testing a Trained Agent

To watch a trained agent play:

```bash
python test_model.py --rom_path path/to/tetris.gb --checkpoint data/agents/best_agent_gen50.pt --render
```

To record gameplay as GIF:

```bash
python test_model.py --rom_path path/to/tetris.gb --checkpoint data/agents/best_agent_gen50.pt --render --record --save_gif
```

## Key Components

### CNN Model Architecture

The project provides two CNN model architectures:
- **Standard CNN**: Basic convolutional network for processing game screens
- **Dueling CNN**: Split into value and advantage streams for more stable learning

### Hybrid DQN Agent

The DQN agent features:
- Experience replay buffer for stable learning
- Target network for stable Q-value estimation
- Epsilon-greedy exploration policy
- Gradient clipping for stable training
- Compatibility with both evolution and gradient-based learning

### Hybrid Training Approach

Our hybrid training approach:
- Maintains a population of agents that play and learn independently
- Collects experiences during gameplay for gradient-based training
- Performs backpropagation updates after each episode
- Evaluates agents based on game performance
- Preserves the best-performing agents (elites)
- Generates new agents through mutations to explore parameter space
- Uses Apple Silicon MPS acceleration for improved performance on M-series Macs
- Supports parallel multi-process training for faster iteration

### Parallel Training

The parallel training functionality:
- Evaluates multiple agents simultaneously in separate processes
- Creates independent PyBoy instances for each agent
- Significantly speeds up training time, especially on multi-core systems
- Automatically distributes workload based on available CPUs
- Consolidates results for population evolution

### Tetris Environment

The PyBoy environment wrapper:
- Handles game state extraction and preprocessing
- Provides a reinforcement learning interface (similar to OpenAI Gym)
- Accurately extracts score from WRAM using proper BCD encoding
- Calculates rewards based on game score
- Manages game lifecycle (reset, step, close)

## Hardware Acceleration

The project supports:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon (M1/M2/M3 chips)
- CPU fallback for other systems

## Configuration

Hyperparameters can be tuned in `configs/default_config.py` or passed as command-line arguments to `main.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [PyBoy](https://github.com/Baekalfen/PyBoy): Game Boy emulator in Python
- [PyTorch](https://pytorch.org/): Deep learning framework
- [Tetris](https://en.wikipedia.org/wiki/Tetris): The classic game by Alexey Pajitnov
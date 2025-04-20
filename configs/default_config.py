"""
Default configuration for PyBoy Tetris CNN
"""

# Environment configuration
ENV_CONFIG = {
    'headless': False,          # Run without visualization during training
    'down_sample': 2,           # Downsample factor for screen
    'frameskip': 15,            # Number of frames to skip between observations
}

# Neural network configuration
MODEL_CONFIG = {
    'dueling': True,            # Use dueling network architecture
    'input_channels': 3,        # RGB channels
}

# DQN Agent configuration
AGENT_CONFIG = {
    'learning_rate': 0.0001,    # Learning rate for optimizer
    'gamma': 0.99,              # Discount factor
    'buffer_size': 10000,       # Replay buffer capacity
    'batch_size': 32,           # Batch size for training
    'target_update_freq': 1000, # Update target network every n steps
    'epsilon_start': 1.0,       # Initial exploration rate
    'epsilon_end': 0.01,        # Final exploration rate
    'epsilon_decay': 10000,     # Exploration decay steps
}

# Population configuration
POPULATION_CONFIG = {
    'population_size': 10,      # Number of agents in population
    'mutation_rate': 0.05,      # Probability of mutating each parameter
    'mutation_scale': 0.1,      # Scale of mutations
    'elite_count': 2,           # Number of top agents preserved unchanged
    'episodes_per_agent': 3,    # Episodes to evaluate each agent
    'max_steps': 10000,         # Maximum steps per episode
}

# Training configuration
TRAIN_CONFIG = {
    'generations': 100,         # Number of generations to train
    'save_interval': 5,         # Save checkpoint every n generations
}

# Paths and directories
PATHS = {
    'save_dir': 'data/agents',  # Directory to save agent checkpoints
    'log_dir': 'data/logs',     # Directory for tensorboard logs
}
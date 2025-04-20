import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import copy

from models.cnn_model import TetrisCNN, DuelingTetrisCNN
from utils.device_utils import get_device


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for playing Tetris
    """
    def __init__(self, state_shape, num_actions, dueling=True, learning_rate=0.0001, 
                 gamma=0.99, buffer_size=10000, batch_size=32, target_update_freq=1000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=10000):
        """
        Initialize the DQN Agent
        
        Args:
            state_shape (tuple): Shape of the state input (C, H, W)
            num_actions (int): Number of possible actions
            dueling (bool): Whether to use dueling DQN architecture
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor for future rewards
            buffer_size (int): Capacity of replay buffer
            batch_size (int): Batch size for training
            target_update_freq (int): Frequency of target network updates
            epsilon_start (float): Initial exploration probability
            epsilon_end (float): Final exploration probability
            epsilon_decay (int): Decay steps for exploration probability
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.dueling = dueling
        self.device = get_device()
        
        # Create networks
        input_channels, input_height, input_width = state_shape
        model_class = DuelingTetrisCNN if dueling else TetrisCNN
        
        self.policy_net = model_class(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            num_actions=num_actions
        ).to(self.device)
        self.target_net = model_class(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            num_actions=num_actions
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for evaluation
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon_start = epsilon_start  # Starting exploration rate
        self.epsilon_end = epsilon_end  # Final exploration rate
        self.epsilon_decay = epsilon_decay  # Decay steps
        self.target_update_freq = target_update_freq  # Update target net every n steps
        self.batch_size = batch_size
        
        # Create replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize counters
        self.steps = 0
        self.episodes = 0
        
    def get_epsilon(self):
        """Calculate current epsilon value based on decay schedule"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 np.exp(-1.0 * self.steps / self.epsilon_decay)
        return max(self.epsilon_end, epsilon)
        
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        epsilon = self.get_epsilon()
        
        if random.random() < epsilon:
            # Random action (exploration)
            return random.randint(0, self.num_actions - 1)
        else:
            # Best action according to policy (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def train(self):
        """
        Perform one training step using a random batch from replay buffer
        """
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples for training
            
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            
        # Compute target Q values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        for param in self.policy_net.parameters():
            if param.grad is not None:  # Check if gradients exist before clipping
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network if needed
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
        
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
        self.steps += 1
        
    def save(self, path):
        """Save agent's policy network to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
        print(f"Agent saved to {path}")
        
    def load(self, path):
        """Load agent's policy network from file"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps = checkpoint['steps']
            self.episodes = checkpoint['episodes']
            print(f"Agent loaded from {path}")
        else:
            print(f"No checkpoint found at {path}")
            
    def get_mutated_copy(self, mutation_rate=0.1, mutation_scale=0.1):
        """
        Create a mutated copy of this agent
        
        Args:
            mutation_rate (float): Probability of mutating each parameter
            mutation_scale (float): Scale of mutations
            
        Returns:
            DQNAgent: A new agent with mutated parameters
        """
        # Create a new agent with the same hyperparameters
        new_agent = DQNAgent(
            state_shape=self.state_shape,
            num_actions=self.num_actions,
            dueling=self.dueling,
            gamma=self.gamma,
            buffer_size=len(self.memory.buffer),
            batch_size=self.batch_size,
            target_update_freq=self.target_update_freq,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay=self.epsilon_decay
        )
        
        # Copy weights from this agent
        new_agent.policy_net.load_state_dict(self.policy_net.state_dict())
        new_agent.target_net.load_state_dict(self.target_net.state_dict())
        
        # Apply mutations to policy network
        with torch.no_grad():
            for param in new_agent.policy_net.parameters():
                # Create mutation mask (which parameters to mutate)
                mutation_mask = torch.rand_like(param) < mutation_rate
                # Create mutation values
                mutations = torch.randn_like(param) * mutation_scale
                # Apply mutations
                param.data[mutation_mask] += mutations[mutation_mask]
                
        # Update target network with mutated weights
        new_agent.target_net.load_state_dict(new_agent.policy_net.state_dict())
        
        return new_agent
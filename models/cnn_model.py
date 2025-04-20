import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.device_utils import get_device


class TetrisCNN(nn.Module):
    """
    Convolutional Neural Network for Tetris game playing
    
    Takes game screen as input and outputs action probabilities
    """
    def __init__(self, input_channels=4, input_height=72, input_width=80, num_actions=9):
        """
        Initialize the CNN
        
        Args:
            input_channels (int): Number of color channels in input image
            input_height (int): Height of input image (downsampled)
            input_width (int): Width of input image (downsampled)
            num_actions (int): Number of possible actions
        """
        super(TetrisCNN, self).__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_actions = num_actions
        self.device = get_device()
        
        # Print input dimensions for debugging
        print(f"CNN Init - Input dims: channels={input_channels}, height={input_height}, width={input_width}")
        
        # First convolutional layer - use smaller kernel for better feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions with proper padding
        def conv_output_size(size, kernel_size, stride, padding=0):
            return ((size + 2 * padding - kernel_size) // stride) + 1
            
        h = input_height
        w = input_width
        
        # Make sure we have valid dimensions before proceeding
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid input dimensions: height={h}, width={w}")
            
        # First conv: kernel=5, stride=2, padding=2
        h = conv_output_size(h, 5, 2, 2)
        w = conv_output_size(w, 5, 2, 2)
        print(f"After conv1: h={h}, w={w}")
        
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid dimensions after conv1: height={h}, width={w}")
            
        # Second conv: kernel=4, stride=2, padding=0
        h = conv_output_size(h, 4, 2)
        w = conv_output_size(w, 4, 2)
        print(f"After conv2: h={h}, w={w}")
        
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid dimensions after conv2: height={h}, width={w}")
            
        # Third conv: kernel=3, stride=1, padding=0
        h = conv_output_size(h, 3, 1)
        w = conv_output_size(w, 3, 1)
        print(f"After conv3: h={h}, w={w}")
        
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid dimensions after conv3: height={h}, width={w}")
            
        linear_input_size = h * w * 64
        print(f"Flattened size for fully connected layer: {linear_input_size} (h={h}, w={w})")
        
        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Action probabilities
        """
        # If input is numpy array, convert to tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
            
        # Ensure correct shape (batch_size, channels, height, width)
        if len(x.shape) == 3:  # (height, width, channels) - from PyBoy
            # PyBoy gives us (H, W, C) but PyTorch expects (C, H, W)
            x = x.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
        elif len(x.shape) == 4 and x.shape[1] != self.input_channels:  # (batch, height, width, channels)
            # Permute to (batch, channels, height, width)
            x = x.permute(0, 3, 1, 2)
            
        # Ensure tensor is on the right device
        x = x.to(self.device)
            
        # Print shape after permutation for debugging
        # print(f"Input shape after processing: {x.shape}")
            
        # Forward through convolutional layers
        x = F.relu(self.conv1(x))
        # print(f"After conv1: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")
        x = F.relu(self.conv3(x))
        # print(f"After conv3: {x.shape}")
        
        # Flatten for fully connected layers
        # Use reshape instead of view for better compatibility with TensorBoard
        x = x.reshape(x.size(0), -1)
        # print(f"After flattening: {x.shape}")
        
        # Forward through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation here, will be applied depending on use case
        
        return x
        
    def act(self, state, epsilon=0.0):
        """
        Choose action based on state
        
        Args:
            state: Environment state
            epsilon (float): Probability of random action (exploration)
            
        Returns:
            int: Selected action
        """
        if torch.rand(1).item() < epsilon:
            # Exploration: random action
            return torch.randint(0, self.num_actions, (1,)).item()
        else:
            # Exploitation: best action
            with torch.no_grad():
                q_values = self(state)
                return torch.argmax(q_values).item()
                
    def save(self, path):
        """Save model weights to path"""
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        """Load model weights from path"""
        self.load_state_dict(torch.load(path, map_location=self.device))
        
        
class DuelingTetrisCNN(TetrisCNN):
    """
    Dueling CNN architecture for Tetris
    
    Splits the network into value and advantage streams for better stability
    """
    def __init__(self, input_channels=4, input_height=72, input_width=80, num_actions=9):
        super(DuelingTetrisCNN, self).__init__(input_channels, input_height, input_width, num_actions)
        
        # Calculate size after convolutions with proper padding (same as parent)
        def conv_output_size(size, kernel_size, stride, padding=0):
            return ((size + 2 * padding - kernel_size) // stride) + 1
            
        h = input_height
        w = input_width
        
        # First conv: kernel=5, stride=2, padding=2
        h = conv_output_size(h, 5, 2, 2)
        w = conv_output_size(w, 5, 2, 2)
        
        # Second conv: kernel=4, stride=2, padding=0
        h = conv_output_size(h, 4, 2)
        w = conv_output_size(w, 4, 2)
        
        # Third conv: kernel=3, stride=1, padding=0
        h = conv_output_size(h, 3, 1)
        w = conv_output_size(w, 3, 1)
        
        linear_input_size = h * w * 64
        print(f"Dueling CNN flattened size: {linear_input_size}")
        
        # Replace the final layers with dueling architecture
        self.fc1 = nn.Linear(linear_input_size, 512)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
    def forward(self, x):
        """
        Forward pass through the dueling network
        """
        # If input is numpy array, convert to tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
            
        # Ensure correct shape (batch_size, channels, height, width)
        if len(x.shape) == 3:  # (height, width, channels)
            # PyBoy gives us (H, W, C) but PyTorch expects (C, H, W)
            x = x.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
        elif len(x.shape) == 4:  # (batch, height, width, channels)
            # Permute to (batch, channels, height, width)
            x = x.permute(0, 3, 1, 2)
            
        # Ensure tensor is on the right device
        x = x.to(self.device)
            
        # Forward through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        # Use reshape instead of view for better compatibility with TensorBoard
        x = x.reshape(x.size(0), -1)
        
        # Forward through first fully connected layer
        x = F.relu(self.fc1(x))
        
        # Split into value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
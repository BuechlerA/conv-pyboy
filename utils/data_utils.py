import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io

def preprocess_screen(screen, down_sample=2, normalize=True):
    """
    Preprocess the Game Boy screen for the neural network
    
    Args:
        screen (numpy.ndarray): Raw screen data from PyBoy
        down_sample (int): Factor by which to downsample
        normalize (bool): Whether to normalize pixel values
        
    Returns:
        numpy.ndarray: Processed screen data
    """
    # Convert if needed
    if not isinstance(screen, np.ndarray):
        screen = np.array(screen)
        
    # Down sample if requested
    if down_sample > 1:
        h, w, c = screen.shape
        new_h, new_w = h // down_sample, w // down_sample
        screen = np.array(Image.fromarray(screen).resize((new_w, new_h)))
        
    # Normalize pixel values to [0, 1]
    if normalize and screen.max() > 1.0:
        screen = screen / 255.0
        
    return screen

def create_figure_image(figure, close_after=True):
    """
    Convert a matplotlib figure to an image array
    
    Args:
        figure (Figure): Matplotlib figure
        close_after (bool): Whether to close the figure after conversion
        
    Returns:
        numpy.ndarray: Image as RGB array
    """
    # Save figure to a buffer
    buf = io.BytesIO()
    figure.savefig(buf, format='png', dpi=100)
    
    # Close figure if requested
    if close_after:
        plt.close(figure)
        
    # Convert buffer to image array
    buf.seek(0)
    img = np.array(Image.open(buf))
    
    return img

def plot_q_values(q_values, actions_dict, figsize=(10, 4)):
    """
    Create a bar chart of Q-values for different actions
    
    Args:
        q_values (torch.Tensor or numpy.ndarray): Q-values for each action
        actions_dict (dict): Mapping from action indices to action names
        figsize (tuple): Figure size
        
    Returns:
        Figure: Matplotlib figure
    """
    if isinstance(q_values, torch.Tensor):
        q_values = q_values.detach().cpu().numpy()
        
    # Create figure
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    
    # Action names and values
    action_names = [actions_dict.get(i, f"Action {i}") for i in range(len(q_values))]
    
    # Create bar chart
    bars = ax.bar(action_names, q_values)
    
    # Highlight best action
    best_idx = np.argmax(q_values)
    bars[best_idx].set_color('green')
    
    # Add labels
    ax.set_title('Q-Values by Action')
    ax.set_xlabel('Action')
    ax.set_ylabel('Q-Value')
    
    # Add value labels
    for i, v in enumerate(q_values):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom' if v >= 0 else 'top')
        
    fig.tight_layout()
    return fig
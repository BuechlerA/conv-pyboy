import torch
import numpy as np
from environment.tetris_env import TetrisEnv

def evaluate_agent_process(agent_id, agent_state_dict, state_shape, num_actions, 
                           rom_path, episodes, max_steps, training_steps, dueling=True):
    """
    Function to run in a separate process to evaluate a single agent
    
    Args:
        agent_id: ID of the agent (for logging)
        agent_state_dict: Serialized agent model weights (from CPU)
        state_shape: Shape of the state input
        num_actions: Number of possible actions
        rom_path: Path to the ROM file
        episodes: Number of episodes to evaluate
        max_steps: Max steps per episode
        training_steps: Training steps per episode
        dueling: Whether to use dueling DQN
        
    Returns:
        tuple: (agent_id, avg_score, updated_agent_state_dict, losses)
    """
    # Need to import here to avoid issues with multiprocessing
    from agents.dqn_agent import DQNAgent
    from utils.device_utils import get_device
    
    # Get appropriate device for this process
    device = get_device()
    
    # Create environment instance for this process
    env = TetrisEnv(rom_path=rom_path, headless=True)
    
    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        dueling=dueling
    )
    
    # Load state dict (already moved to CPU in the main process)
    agent.policy_net.load_state_dict(agent_state_dict)
    agent.target_net.load_state_dict(agent_state_dict)
    
    # Evaluate agent
    agent_scores = []
    all_losses = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.add_experience(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            step_count += 1
        
        # Train after episode
        episode_losses = []
        for _ in range(training_steps):
            loss = agent.train()
            if loss > 0:
                episode_losses.append(loss)
        
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            all_losses.append(avg_loss)
            print(f"Process {agent_id} - Episode {episode+1} - Score: {total_reward:.2f}, Avg Loss: {avg_loss:.6f}")
        else:
            print(f"Process {agent_id} - Episode {episode+1} - Score: {total_reward:.2f}, No Training")
        
        agent_scores.append(total_reward)
    
    # Clean up
    env.close()
    
    # Return results - Move state dict back to CPU for transfer across processes
    avg_score = np.mean(agent_scores)
    cpu_state_dict = {k: v.detach().cpu() for k, v in agent.policy_net.state_dict().items()}
    
    return (agent_id, avg_score, cpu_state_dict, all_losses)
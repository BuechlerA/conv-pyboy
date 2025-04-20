import os
import argparse
import time
import torch
import numpy as np
from PIL import Image

from environment.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from utils.device_utils import get_device, print_device_info


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test a trained Tetris agent")
    
    parser.add_argument('--rom_path', type=str, required=True, help='Path to the Tetris ROM file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the agent checkpoint file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', help='Enable visualization')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between frames (seconds)')
    parser.add_argument('--record', action='store_true', help='Record gameplay as images')
    parser.add_argument('--record_dir', type=str, default='data/recordings', help='Directory to save recordings')
    parser.add_argument('--save_gif', action='store_true', help='Save recordings as GIF')
    parser.add_argument('--epsilon', type=float, default=0.0, help='Exploration rate (0 for fully greedy)')
    
    return parser.parse_args()


def main():
    """Test a trained agent"""
    args = parse_args()
    
    # Get appropriate device for computation
    device = get_device()
    print_device_info()
    
    # Configure recording
    if args.record:
        os.makedirs(args.record_dir, exist_ok=True)
    
    # Create environment
    env = TetrisEnv(rom_path=args.rom_path, headless=(not args.render))
    
    # Get observation shape
    initial_state = env.reset()
    # PyBoy gives us (H, W, C), but we need (C, H, W) for PyTorch
    state_shape = (initial_state.shape[2], initial_state.shape[0], initial_state.shape[1])
    num_actions = len(env.ACTIONS)
    
    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        dueling=True
    )
    
    # Load checkpoint
    agent.load(args.checkpoint)
    print(f"Loaded agent from {args.checkpoint}")
    
    # Run test episodes
    total_scores = []
    
    for episode in range(1, args.episodes + 1):
        print(f"\nEpisode {episode}/{args.episodes}")
        
        state = env.reset()
        frames = []
        total_reward = 0
        step = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state) if random.random() > args.epsilon else random.randint(0, num_actions - 1)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update total reward
            total_reward += reward
            
            # For recording
            if args.record:
                # Convert state to displayable format (RGB uint8)
                frame = (env.pyboy.screen.screen_ndarray() * 255).astype(np.uint8)
                frames.append(Image.fromarray(frame))
            
            # Add delay for visualization
            if args.render and args.delay > 0:
                time.sleep(args.delay)
                
            # Update state
            state = next_state
            step += 1
            
            # Print progress
            if step % 100 == 0:
                print(f"Step {step}, Current Score: {info['score']}")
                
        print(f"Episode {episode} - Score: {total_reward:.2f}, Steps: {step}")
        total_scores.append(total_reward)
        
        # Save recording
        if args.record and frames:
            recording_path = os.path.join(args.record_dir, f"episode_{episode}")
            
            # Save as individual frames
            os.makedirs(recording_path, exist_ok=True)
            for i, frame in enumerate(frames):
                frame.save(os.path.join(recording_path, f"frame_{i:04d}.png"))
                
            # Save as GIF
            if args.save_gif:
                gif_path = os.path.join(args.record_dir, f"episode_{episode}.gif")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    optimize=True,
                    duration=args.delay * 1000,  # milliseconds
                    loop=0
                )
                print(f"Saved GIF to {gif_path}")
    
    # Print summary
    print("\nTest Results:")
    print(f"Episodes: {args.episodes}")
    print(f"Average Score: {np.mean(total_scores):.2f}")
    print(f"Max Score: {np.max(total_scores):.2f}")
    print(f"Min Score: {np.min(total_scores):.2f}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    import random
    main()
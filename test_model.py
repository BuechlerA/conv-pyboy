import os
import sys
import argparse
import time
import torch
import numpy as np
import logging
from datetime import datetime
from PIL import Image

from environment.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from utils.device_utils import get_device, print_device_info
from utils.logging_utils import setup_logging, setup_exception_logging


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
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                      default='INFO', help='Logging level')
    parser.add_argument('--log_dir', type=str, default='data/logs', help='Directory for logs')
    
    return parser.parse_args()


def main():
    """Test a trained agent"""
    args = parse_args()
    
    # Set up logging
    log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, "test_logs")
    log_file = f"test_model_{log_timestamp}.log"
    
    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level)
    
    # Initialize logging
    setup_logging(log_dir=log_dir, log_file=log_file, log_level=log_level)
    setup_exception_logging()
    
    # Log basic info
    logging.info("Starting ConvPyBoy testing session")
    logging.info(f"Command line arguments: {args}")
    
    try:
        # Get appropriate device for computation
        device = get_device()
        print_device_info()
        logging.info(f"Using device: {device}")
        
        # Configure recording
        if args.record:
            os.makedirs(args.record_dir, exist_ok=True)
            logging.info(f"Recording enabled. Frames will be saved to {args.record_dir}")
        
        # Create environment
        logging.info(f"Initializing Tetris environment with ROM: {args.rom_path}")
        env = TetrisEnv(rom_path=args.rom_path, headless=(not args.render))
        
        # Get observation shape
        initial_state = env.reset()
        # PyBoy gives us (H, W, C), but we need (C, H, W) for PyTorch
        state_shape = (initial_state.shape[2], initial_state.shape[0], initial_state.shape[1])
        num_actions = len(env.ACTIONS)
        logging.info(f"Environment initialized. State shape: {initial_state.shape}, Actions: {num_actions}")
        
        # Create agent
        logging.info("Creating DQN agent")
        agent = DQNAgent(
            state_shape=state_shape,
            num_actions=num_actions,
            dueling=True
        )
        
        # Load checkpoint
        logging.info(f"Loading agent checkpoint from {args.checkpoint}")
        agent.load(args.checkpoint)
        
        # Run test episodes
        total_scores = []
        
        for episode in range(1, args.episodes + 1):
            logging.info(f"Starting episode {episode}/{args.episodes}")
            
            state = env.reset()
            frames = []
            total_reward = 0
            step = 0
            done = False
            
            while not done:
                try:
                    # Select action
                    if random.random() > args.epsilon:
                        action = agent.select_action(state)
                    else:
                        action = random.randint(0, num_actions - 1)
                        logging.debug(f"Random exploration action: {action}")
                    
                    # Take action
                    next_state, reward, done, info = env.step(action)
                    
                    # Update total reward
                    total_reward += reward
                    
                    # For recording
                    if args.record:
                        try:
                            # Convert state to displayable format (RGB uint8)
                            frame = (env.pyboy.screen.screen_ndarray() * 255).astype(np.uint8)
                            frames.append(Image.fromarray(frame))
                        except Exception as e:
                            logging.error(f"Error recording frame: {str(e)}")
                    
                    # Add delay for visualization
                    if args.render and args.delay > 0:
                        time.sleep(args.delay)
                        
                    # Update state
                    state = next_state
                    step += 1
                    
                    # Log progress
                    if step % 100 == 0:
                        logging.info(f"Episode {episode}, Step {step}, Current Score: {info['score']}")
                
                except Exception as e:
                    logging.error(f"Error during episode step: {str(e)}", exc_info=True)
                    # If something goes wrong during a step, try to continue
                    if step > 0:
                        logging.warning("Attempting to continue episode")
                        continue
                    else:
                        # If we can't even complete one step, break
                        logging.error("Could not complete episode, breaking")
                        break
                        
            logging.info(f"Episode {episode} finished - Score: {total_reward:.2f}, Steps: {step}")
            total_scores.append(total_reward)
            
            # Save recording
            if args.record and frames:
                try:
                    recording_path = os.path.join(args.record_dir, f"episode_{episode}")
                    
                    # Save as individual frames
                    os.makedirs(recording_path, exist_ok=True)
                    logging.info(f"Saving {len(frames)} frames to {recording_path}")
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
                        logging.info(f"Saved GIF to {gif_path}")
                except Exception as e:
                    logging.error(f"Error saving recordings: {str(e)}", exc_info=True)
        
        # Log summary
        logging.info("\nTest Results:")
        logging.info(f"Episodes: {args.episodes}")
        logging.info(f"Average Score: {np.mean(total_scores):.2f}")
        logging.info(f"Max Score: {np.max(total_scores):.2f}")
        logging.info(f"Min Score: {np.min(total_scores):.2f}")
        
        # Close environment
        env.close()
        
    except Exception as e:
        logging.critical(f"Fatal error in testing process: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    import random
    try:
        main()
    except Exception as e:
        logging.critical(f"Unhandled exception in main: {str(e)}", exc_info=True)
        sys.exit(1)
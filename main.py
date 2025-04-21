import os
import argparse
import torch
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.loggers import TensorBoardLogger

from environment.tetris_env import TetrisEnv
from agents.population import Population
from utils.device_utils import get_device, print_device_info
from utils.logging_utils import setup_logging, setup_exception_logging


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a PyBoy Tetris CNN using evolutionary reinforcement learning")
    
    parser.add_argument('--rom_path', type=str, required=True, help='Path to the Tetris ROM file')
    parser.add_argument('--population_size', type=int, default=10, help='Number of agents in the population')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations to train')
    parser.add_argument('--mutation_rate', type=float, default=0.05, help='Probability of mutating each parameter')
    parser.add_argument('--mutation_scale', type=float, default=0.1, help='Scale of mutations')
    parser.add_argument('--elite_count', type=int, default=2, help='Number of top agents to preserve unchanged')
    parser.add_argument('--episodes_per_agent', type=int, default=3, help='Number of episodes to evaluate each agent')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum steps per episode')
    parser.add_argument('--save_dir', type=str, default='data/agents', help='Directory to save agent checkpoints')
    parser.add_argument('--log_dir', type=str, default='data/logs', help='Directory for tensorboard logs')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no visualization)')
    parser.add_argument('--load_checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--training_steps', type=int, default=32, help='Number of gradient training steps per episode')
    parser.add_argument('--parallel', action='store_true', help='Use parallel training')
    parser.add_argument('--num_processes', type=int, default=None, 
                        help='Number of parallel processes (defaults to min of CPU count and population size)')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help='Logging level')
    
    return parser.parse_args()


def visualize_scores(scores_history, generation, save_path=None):
    """Visualize scores over generations"""
    plt.figure(figsize=(10, 6))
    
    # Plot best, average, and worst scores
    best_scores = [np.max(scores) for scores in scores_history]
    avg_scores = [np.mean(scores) for scores in scores_history]
    worst_scores = [np.min(scores) for scores in scores_history]
    
    x = list(range(1, len(scores_history) + 1))
    
    plt.plot(x, best_scores, label='Best Score', marker='o')
    plt.plot(x, avg_scores, label='Average Score', marker='x')
    plt.plot(x, worst_scores, label='Worst Score', marker='.')
    
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title(f'Score Evolution over Generations (Current: {generation})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        logging.debug(f"Score plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    """Main training loop"""
    args = parse_args()
    
    # Set up logging
    log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, "logs")
    log_file = f"conv_pyboy_run_{log_timestamp}.log"
    
    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level)
    
    # Initialize logging
    setup_logging(log_dir=log_dir, log_file=log_file, log_level=log_level)
    setup_exception_logging()
    
    # Log basic info
    logging.info("Starting ConvPyBoy training session")
    logging.info(f"Command line arguments: {args}")
    
    try:
        # Create directories
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'plots'), exist_ok=True)
        
        # Get the device for computation
        device = get_device()
        print_device_info()
        logging.info(f"Using device: {device}")
        
        # Setup Lightning TensorBoard logger
        tensorboard_version = datetime.now().strftime("%Y%m%d-%H%M%S")
        logger = TensorBoardLogger(save_dir=args.log_dir, name="conv_pyboy", version=tensorboard_version)
        writer = logger.experiment
        # Log hyperparameters to TensorBoard
        logger.log_hyperparams(vars(args))
        logging.info(f"TensorBoard logs will be saved to {logger.log_dir}")
        
        # Create environment
        logging.info(f"Initializing Tetris environment with ROM: {args.rom_path}")
        env = TetrisEnv(rom_path=args.rom_path, headless=args.headless)
        
        # Get observation space shape
        initial_state = env.reset()
        # PyBoy gives us (H, W, C) format, but we need (C, H, W) for PyTorch
        state_shape = (initial_state.shape[2], initial_state.shape[0], initial_state.shape[1])
        
        # Number of possible actions
        num_actions = len(env.ACTIONS)
        
        logging.info(f"State shape: {initial_state.shape}, PyTorch state shape: {state_shape}, Number of actions: {num_actions}")
        
        # Create population
        logging.info(f"Creating population with {args.population_size} agents")
        population = Population(
            state_shape=state_shape,
            num_actions=num_actions,
            population_size=args.population_size,
            mutation_rate=args.mutation_rate,
            mutation_scale=args.mutation_scale,
            elite_count=args.elite_count,
            save_dir=args.save_dir,
            training_steps_per_episode=args.training_steps  # Add training steps parameter
        )
        
        # Load checkpoint if specified
        if args.load_checkpoint:
            logging.info(f"Loading checkpoint from {args.load_checkpoint}")
            population.load_checkpoint(args.load_checkpoint)
        
        # Track scores across generations
        scores_history = []
        
        # Main training loop
        for generation in range(1, args.generations + 1):
            logging.info(f"Starting Generation {generation}/{args.generations}")
            
            try:
                # Evaluate fitness of all agents - choose between sequential and parallel
                if args.parallel:
                    logging.info("Using parallel training mode")
                    fitness_scores = population.evaluate_fitness_parallel(
                        env=env,  # Main environment (for reference)
                        episodes_per_agent=args.episodes_per_agent,
                        max_steps=args.max_steps,
                        writer=writer,
                        generation=generation,
                        num_processes=args.num_processes,
                        rom_path=args.rom_path
                    )
                else:
                    # Original sequential evaluation
                    logging.info("Using sequential training mode")
                    fitness_scores = population.evaluate_fitness(
                        env=env,
                        episodes_per_agent=args.episodes_per_agent,
                        max_steps=args.max_steps,
                        writer=writer,  # Pass TensorBoard writer to log training metrics
                        generation=generation  # Pass current generation number
                    )
                
                # Record scores
                scores_history.append(fitness_scores)
                best_score = np.max(fitness_scores)
                avg_score = np.mean(fitness_scores)
                worst_score = np.min(fitness_scores)
                
                # Log score statistics
                logging.info(f"Generation {generation} scores - Best: {best_score:.2f}, Average: {avg_score:.2f}, Worst: {worst_score:.2f}")
                
                # Log to tensorboard
                writer.add_scalar('Scores/Best', best_score, generation)
                writer.add_scalar('Scores/Average', avg_score, generation)
                writer.add_scalar('Scores/Worst', worst_score, generation)
                
                # Visualize and save plot
                plot_path = os.path.join(args.log_dir, 'plots', f'scores_gen{generation}.png')
                visualize_scores(scores_history, generation, save_path=plot_path)
                
                # Evolve population
                logging.info(f"Evolving population for generation {generation}")
                population.evolve()
                
                # Log best agent architecture to tensorboard
                try:
                    best_agent = population.get_best_agent()
                    
                    # Create a deep copy of the model for visualization
                    import copy
                    model_for_vis = copy.deepcopy(best_agent.policy_net)
                    
                    # Move model to CPU and ensure it's using standard FloatTensor
                    model_for_vis = model_for_vis.to("cpu")
                    
                    # Process input data: create proper tensor shape, move to CPU
                    if len(initial_state.shape) == 3:  # (H, W, C)
                        # Convert to (C, H, W) format expected by PyTorch
                        # Fix: Ensure correct permutation from (H, W, C) to (C, H, W)
                        input_tensor = torch.FloatTensor(initial_state).permute(2, 0, 1)
                        
                        # Log the tensor shape for debugging
                        logging.debug(f"Input tensor shape after permutation: {input_tensor.shape}")
                        
                        # Double check shape matches expected model input
                        channels, height, width = input_tensor.shape
                        logging.debug(f"Expected model input: channels={state_shape[0]}, height={state_shape[1]}, width={state_shape[2]}")
                        
                        # Explicitly verify dimensions match what's expected by the model
                        if channels != state_shape[0] or height != state_shape[1] or width != state_shape[2]:
                            logging.warning(f"Tensor shape mismatch: got ({channels}, {height}, {width}), expected {state_shape}")
                            # Force the correct shape if needed
                            input_tensor = input_tensor.reshape(state_shape[0], state_shape[1], state_shape[2])
                    else:
                        input_tensor = torch.FloatTensor(initial_state)
                    
                    # Add batch dimension and ensure on CPU
                    input_tensor = input_tensor.unsqueeze(0).to("cpu")
                    
                    # Log the final input shape before passing to the model
                    logging.debug(f"Final input tensor shape with batch dimension: {input_tensor.shape}")
                    
                    # Log graph to TensorBoard, with error handling
                    try:
                        writer.add_graph(model_for_vis, input_tensor)
                        logging.info("TensorBoard graph visualization saved successfully")
                    except Exception as e:
                        logging.warning(f"Error generating TensorBoard graph: {str(e)}")
                        logging.warning("Skipping graph visualization and continuing training")
                        
                except Exception as e:
                    logging.error(f"Error during TensorBoard graph visualization: {str(e)}", exc_info=True)
                    logging.info("Continuing training process")
                
                # Save progress
                if generation % 5 == 0:
                    checkpoint_path = os.path.join(args.save_dir, f'checkpoint_gen{generation}.pt')
                    best_agent.save(checkpoint_path)
                    logging.info(f"Checkpoint saved to {checkpoint_path}")
                
            except Exception as e:
                logging.error(f"Error during generation {generation}: {str(e)}", exc_info=True)
                logging.warning("Attempting to continue with next generation")
                continue
        
        # Close environment and writer
        env.close()
        writer.close()
        
        logging.info("\nTraining completed successfully!")
        logging.info(f"Best score achieved: {population.best_score_ever:.2f}")
        logging.info(f"Checkpoints saved to {args.save_dir}")
        logging.info(f"Logs saved to {logger.log_dir}")
    
    except Exception as e:
        logging.critical(f"Fatal error in main process: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    import multiprocessing as mp  # Ensure spawn method for CUDA in subprocesses
    mp.set_start_method('spawn', force=True)
    try:
        main()
    except Exception as e:
        logging.critical(f"Fatal error in main process: {str(e)}", exc_info=True)
        # Exit with error code to indicate failure (useful for scripts)
        sys.exit(1)
import os
import argparse
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from environment.tetris_env import TetrisEnv
from agents.population import Population
from utils.device_utils import get_device, print_device_info


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
    else:
        plt.show()
    plt.close()


def main():
    """Main training loop"""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'plots'), exist_ok=True)
    
    # Get the device for computation
    device = get_device()
    print_device_info()
    
    # Setup tensorboard
    log_dir = os.path.join(args.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create environment
    env = TetrisEnv(rom_path=args.rom_path, headless=args.headless)
    
    # Get observation space shape
    initial_state = env.reset()
    # PyBoy gives us (H, W, C) format, but we need (C, H, W) for PyTorch
    state_shape = (initial_state.shape[2], initial_state.shape[0], initial_state.shape[1])
    
    # Number of possible actions
    num_actions = len(env.ACTIONS)
    
    print(f"State shape: {initial_state.shape}, PyTorch state shape: {state_shape}, Number of actions: {num_actions}")
    
    # Create population
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
        population.load_checkpoint(args.load_checkpoint)
        print(f"Loaded checkpoint from {args.load_checkpoint}")
    
    # Track scores across generations
    scores_history = []
    
    # Main training loop
    for generation in range(1, args.generations + 1):
        print(f"\n{'='*40}")
        print(f"Generation {generation}/{args.generations}")
        print(f"{'='*40}")
        
        # Evaluate fitness of all agents - choose between sequential and parallel
        if args.parallel:
            print("Using parallel training mode")
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
            fitness_scores = population.evaluate_fitness(
                env=env,
                episodes_per_agent=args.episodes_per_agent,
                max_steps=args.max_steps,
                writer=writer,  # Pass TensorBoard writer to log training metrics
                generation=generation  # Pass current generation number
            )
        
        # Record scores
        scores_history.append(fitness_scores)
        
        # Log to tensorboard
        writer.add_scalar('Scores/Best', np.max(fitness_scores), generation)
        writer.add_scalar('Scores/Average', np.mean(fitness_scores), generation)
        writer.add_scalar('Scores/Worst', np.min(fitness_scores), generation)
        
        # Visualize and save plot
        plot_path = os.path.join(args.log_dir, 'plots', f'scores_gen{generation}.png')
        visualize_scores(scores_history, generation, save_path=plot_path)
        
        # Evolve population
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
                channels = initial_state.shape[2]
                height = initial_state.shape[0]
                width = initial_state.shape[1]
                input_tensor = torch.FloatTensor(initial_state).permute(2, 0, 1)
            else:
                input_tensor = torch.FloatTensor(initial_state)
            
            # Add batch dimension and ensure on CPU
            input_tensor = input_tensor.unsqueeze(0).to("cpu")
            
            # Log graph to TensorBoard, with error handling
            try:
                writer.add_graph(model_for_vis, input_tensor)
                print("TensorBoard graph visualization saved successfully")
            except Exception as e:
                print(f"Error generating TensorBoard graph: {str(e)}")
                print("Skipping graph visualization and continuing training")
                
        except Exception as e:
            print(f"Error during TensorBoard graph visualization: {str(e)}")
            print("Continuing training process")
        
        # Save progress
        if generation % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_gen{generation}.pt')
            best_agent.save(checkpoint_path)
            
    # Close environment and writer
    env.close()
    writer.close()
    
    print("\nTraining completed!")
    print(f"Best score achieved: {population.best_score_ever:.2f}")
    print(f"Checkpoints saved to {args.save_dir}")
    print(f"Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
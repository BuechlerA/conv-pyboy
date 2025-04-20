import os
import torch
import numpy as np
import multiprocessing as mp
from .dqn_agent import DQNAgent
from .worker import evaluate_agent_process
from utils.device_utils import get_device

class Population:
    """
    Population of agents implementing evolutionary algorithm
    
    Manages multiple agents playing the game and evolves them based on their performance
    """
    def __init__(self, state_shape, num_actions, population_size=10, 
                 mutation_rate=0.1, mutation_scale=0.1, 
                 elite_count=1, save_dir="data/agents",
                 training_steps_per_episode=10):  # Added training steps parameter
        """
        Initialize a population of agents
        
        Args:
            state_shape (tuple): Shape of the state input (C, H, W)
            num_actions (int): Number of possible actions
            population_size (int): Number of agents in the population
            mutation_rate (float): Probability of mutating each parameter
            mutation_scale (float): Scale of mutations
            elite_count (int): Number of top agents to preserve unchanged
            save_dir (str): Directory to save agent checkpoints
            training_steps_per_episode (int): Number of gradient updates to perform after each episode
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elite_count = elite_count
        self.save_dir = save_dir
        self.device = get_device()
        self.training_steps_per_episode = training_steps_per_episode  # Store training steps
        
        # Create directory for saving agents
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize population with random agents
        self.agents = [
            DQNAgent(
                state_shape=state_shape,
                num_actions=num_actions,
                dueling=True
            )
            for _ in range(population_size)
        ]
        
        # Initialize fitness scores
        self.fitness_scores = [0] * population_size
        self.best_score_ever = 0
        self.generation = 0
        
    def evaluate_fitness(self, env, episodes_per_agent=3, max_steps=10000, writer=None, generation=0):
        """
        Evaluate fitness of all agents in the population
        
        Args:
            env: Game environment
            episodes_per_agent (int): Number of episodes to evaluate each agent
            max_steps (int): Maximum number of steps per episode
            writer: TensorBoard writer for logging (optional)
            generation: Current generation number for TensorBoard logging
            
        Returns:
            list: Fitness scores for all agents
        """
        self.fitness_scores = []
        
        for i, agent in enumerate(self.agents):
            # Evaluate agent over multiple episodes
            agent_scores = []
            total_losses = []
            
            for episode in range(episodes_per_agent):
                state = env.reset()
                done = False
                total_reward = 0
                step_count = 0
                episode_losses = []
                
                while not done and step_count < max_steps:
                    # Select action
                    action = agent.select_action(state)
                    
                    # Execute action
                    next_state, reward, done, info = env.step(action)
                    
                    # Store experience in replay buffer
                    agent.add_experience(state, action, reward, next_state, done)
                    
                    # Record reward
                    total_reward += reward
                    
                    # Update state
                    state = next_state
                    step_count += 1
                
                # Train the network after each episode
                training_batch_losses = []
                for train_step in range(self.training_steps_per_episode):
                    loss = agent.train()  # This will now use GPU for backpropagation
                    if loss > 0:  # Only record non-zero losses
                        training_batch_losses.append(loss)
                        
                    # Log training progress to TensorBoard every 10 steps
                    if writer is not None and train_step % 10 == 0 and loss > 0:
                        global_step = (generation * self.population_size * episodes_per_agent * self.training_steps_per_episode) + \
                                      (i * episodes_per_agent * self.training_steps_per_episode) + \
                                      (episode * self.training_steps_per_episode) + train_step
                        writer.add_scalar(f'Training/Agent_{i+1}/Loss', loss, global_step)
                
                if training_batch_losses:  # If we actually performed training
                    avg_loss = np.mean(training_batch_losses)
                    episode_losses.append(avg_loss)
                    print(f"  Episode {episode+1} - Score: {total_reward:.2f}, Avg Loss: {avg_loss:.6f}")
                    
                    # Log episode metrics to TensorBoard
                    if writer is not None:
                        writer.add_scalar(f'Training/Agent_{i+1}/Episode_Loss', avg_loss, generation * episodes_per_agent + episode)
                        writer.add_scalar(f'Training/Agent_{i+1}/Episode_Score', total_reward, generation * episodes_per_agent + episode)
                else:
                    print(f"  Episode {episode+1} - Score: {total_reward:.2f}, No Training (insufficient samples)")
                
                agent_scores.append(total_reward)
                
                # Add episode losses to total losses for this agent
                if episode_losses:
                    total_losses.extend(episode_losses)
            
            # Calculate average score
            avg_score = np.mean(agent_scores)
            self.fitness_scores.append(avg_score)
            
            # Print information with training details
            if total_losses:
                avg_training_loss = np.mean(total_losses)
                print(f"Agent {i+1}/{len(self.agents)} - Avg Score: {avg_score:.2f}, Avg Training Loss: {avg_training_loss:.6f}")
                
                # Log overall agent performance
                if writer is not None:
                    writer.add_scalar(f'Performance/Agent_{i+1}/Avg_Score', avg_score, generation)
                    writer.add_scalar(f'Performance/Agent_{i+1}/Avg_Loss', avg_training_loss, generation)
            else:
                print(f"Agent {i+1}/{len(self.agents)} - Avg Score: {avg_score:.2f}, No Training")
            
        return self.fitness_scores
        
    def evaluate_fitness_parallel(self, env, episodes_per_agent=3, max_steps=10000, writer=None, 
                                  generation=0, num_processes=None, rom_path=None):
        """
        Parallel evaluation of fitness for all agents
        
        Args:
            env: Game environment (for reference only, not used in parallel mode)
            episodes_per_agent: Number of episodes to evaluate each agent
            max_steps: Maximum number of steps per episode
            writer: TensorBoard writer for logging (optional)
            generation: Current generation number for logging
            num_processes: Number of parallel processes to use (defaults to min of CPU count and population size)
            rom_path: Path to ROM file for creating environments in worker processes
            
        Returns:
            list: Fitness scores for all agents
        """
        # Determine number of processes to use
        if num_processes is None:
            num_processes = min(mp.cpu_count(), self.population_size)
        
        # We need the ROM path for each worker process
        if rom_path is None:
            rom_path = env.rom_path
            
        print(f"Starting parallel evaluation with {num_processes} processes")
            
        # Create a pool of worker processes
        with mp.Pool(processes=num_processes) as pool:
            # Prepare tasks for each agent
            tasks = []
            for i, agent in enumerate(self.agents):
                # Get agent state dict (serializable)
                # Move state dict to CPU to avoid MPS/CUDA tensor sharing issues
                cpu_state_dict = {k: v.detach().cpu() for k, v in agent.policy_net.state_dict().items()}
                
                # Create task
                task = (
                    i,  # agent_id
                    cpu_state_dict,
                    self.state_shape,
                    self.num_actions,
                    rom_path,
                    episodes_per_agent,
                    max_steps,
                    self.training_steps_per_episode,
                    agent.dueling
                )
                tasks.append(task)
            
            # Execute tasks in parallel
            results = pool.starmap(evaluate_agent_process, tasks)
        
        # Process results
        self.fitness_scores = []
        for agent_id, avg_score, updated_state_dict, losses in results:
            # Update agent's weights
            self.agents[agent_id].policy_net.load_state_dict(updated_state_dict)
            self.agents[agent_id].target_net.load_state_dict(updated_state_dict)
            
            # Record fitness score
            self.fitness_scores.append(avg_score)
            
            # Log to TensorBoard if available
            if writer is not None and losses:
                avg_training_loss = np.mean(losses)
                writer.add_scalar(f'Performance/Agent_{agent_id+1}/Avg_Score', avg_score, generation)
                writer.add_scalar(f'Performance/Agent_{agent_id+1}/Avg_Loss', avg_training_loss, generation)
                
            # Print performance
            if losses:
                avg_loss = np.mean(losses)
                print(f"Agent {agent_id+1}/{len(self.agents)} - Avg Score: {avg_score:.2f}, Avg Loss: {avg_loss:.6f}")
            else:
                print(f"Agent {agent_id+1}/{len(self.agents)} - Avg Score: {avg_score:.2f}, No Training")
        
        return self.fitness_scores
        
    def evolve(self):
        """
        Evolve the population based on fitness scores
        
        Creates a new generation with elites and mutated copies of the best agents
        """
        self.generation += 1
        print(f"\nEvolution - Generation {self.generation}")
        print(f"Fitness scores: {[f'{score:.2f}' for score in self.fitness_scores]}")
        
        # Get indices of agents sorted by fitness (descending)
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # Get the best score from this generation
        best_score = self.fitness_scores[sorted_indices[0]]
        if best_score > self.best_score_ever:
            self.best_score_ever = best_score
            
        print(f"Best agent score: {best_score:.2f}")
        print(f"Best score ever: {self.best_score_ever:.2f}")
        
        # Save the best agent
        self.agents[sorted_indices[0]].save(
            os.path.join(self.save_dir, f"best_agent_gen{self.generation}.pt")
        )
        
        # Create new population
        new_agents = []
        
        # Add elite agents unchanged
        for i in range(min(self.elite_count, len(self.agents))):
            new_agents.append(self.agents[sorted_indices[i]])
            
        # Fill the rest with mutations of top agents
        while len(new_agents) < self.population_size:
            # Select parent agent (from top half of population)
            parent_idx = sorted_indices[np.random.randint(0, max(1, len(self.agents) // 2))]
            parent = self.agents[parent_idx]
            
            # Create mutated copy
            child = parent.get_mutated_copy(
                mutation_rate=self.mutation_rate,
                mutation_scale=self.mutation_scale
            )
            
            new_agents.append(child)
            
        # Replace old population
        self.agents = new_agents
        
        print(f"Population evolved: {len(self.agents)} agents ready for next generation")
        
    def get_best_agent(self):
        """Get the agent with the highest fitness score"""
        best_idx = np.argmax(self.fitness_scores)
        return self.agents[best_idx]
        
    def load_checkpoint(self, checkpoint_path):
        """Load the best agent from a checkpoint"""
        best_agent = self.agents[0]
        best_agent.load(checkpoint_path)
        
        # Clone best agent to entire population with mutations
        self.agents = [best_agent]
        
        # Add mutations to create diverse population
        for _ in range(1, self.population_size):
            mutated = best_agent.get_mutated_copy(
                mutation_rate=self.mutation_rate,
                mutation_scale=self.mutation_scale
            )
            self.agents.append(mutated)
            
        print(f"Loaded checkpoint {checkpoint_path} into population")
        return best_agent
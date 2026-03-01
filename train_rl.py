"""
Main Training Script for EvolveRL
Implements:
- Multi-Agent Q-Learning training loop
- Population dynamics tracking
- Lotka-Volterra comparison
- Visualization and data export
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import sys

# Add uploads directory to path
sys.path.append('/mnt/user-data/uploads')

from rl_environment import PredatorPreyEnv
from q_learning import MultiAgentQLearning


def train_predator_prey(
    num_episodes: int = 1000,
    num_prey: int = 5,
    num_predator: int = 5,
    render_every: int = 100,
    save_dir: str = '/mnt/user-data/outputs',
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995
):
    """
    Train multi-agent Q-learning on predator-prey environment
    
    Args:
        num_episodes: Number of training episodes
        num_prey: Initial number of prey agents
        num_predator: Initial number of predator agents
        render_every: Render environment every N episodes
        save_dir: Directory to save results
        learning_rate: Q-learning alpha parameter
        discount_factor: Q-learning gamma parameter
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Epsilon decay rate per episode
    """
    
    # Create environment
    env = PredatorPreyEnv(
        render_mode=None,  # Set to "human" to visualize
        num_prey=num_prey,
        num_predator=num_predator
    )
    
    # Initialize Q-learning agents
    prey_agents = [f"prey_{i}" for i in range(num_prey)]
    predator_agents = [f"predator_{i}" for i in range(num_predator)]
    
    ql_manager = MultiAgentQLearning(
        prey_agents=prey_agents,
        predator_agents=predator_agents,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay
    )
    
    # Training statistics
    episode_stats = {
        'episode': [],
        'prey_count': [],
        'predator_count': [],
        'avg_prey_reward': [],
        'avg_predator_reward': [],
        'epsilon': [],
        'total_steps': []
    }
    
    # Full population history for Lotka-Volterra comparison
    full_population_history = {
        'prey': [],
        'predator': [],
        'timesteps': []
    }
    
    print("=" * 60)
    print("EVOLVERL - Multi-Agent Reinforcement Learning Training")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Initial Prey: {num_prey}")
    print(f"Initial Predators: {num_predator}")
    print(f"Learning Rate (α): {learning_rate}")
    print(f"Discount Factor (γ): {discount_factor}")
    print("=" * 60)
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        observations, infos = env.reset()
        
        # Store previous observations for learning
        prev_observations = {}
        prev_actions = {}
        
        done = False
        episode_steps = 0
        
        while not done and episode_steps < 1000:
            # Get actions from Q-learning agents
            actions = ql_manager.get_actions(observations, training=True)
            
            # Store for learning
            for agent in observations:
                prev_observations[agent] = observations[agent]
                if agent in actions:
                    prev_actions[agent] = actions[agent]
            
            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Update Q-values for all agents
            for agent in prev_observations:
                if agent in prev_actions and agent in rewards:
                    # Get next observation (use zeros if agent terminated)
                    next_obs = next_observations.get(agent, np.zeros_like(prev_observations[agent]))
                    done_flag = terminations.get(agent, False) or truncations.get(agent, False)
                    
                    ql_manager.update(
                        agent_name=agent,
                        state=prev_observations[agent],
                        action=prev_actions[agent],
                        reward=rewards[agent],
                        next_state=next_obs,
                        done=done_flag
                    )
            
            observations = next_observations
            episode_steps += 1
            
            # Check if episode done
            if len(observations) == 0 or all(truncations.values()):
                done = True
        
        # Decay epsilon
        ql_manager.decay_epsilon()
        
        # Collect episode statistics
        final_prey = sum(1 for a in env.agents if 'prey' in a)
        final_predator = sum(1 for a in env.agents if 'predator' in a)
        stats = ql_manager.get_statistics()
        
        episode_stats['episode'].append(episode)
        episode_stats['prey_count'].append(final_prey)
        episode_stats['predator_count'].append(final_predator)
        episode_stats['avg_prey_reward'].append(stats['avg_prey_reward'])
        episode_stats['avg_predator_reward'].append(stats['avg_predator_reward'])
        episode_stats['epsilon'].append(stats['epsilon'])
        episode_stats['total_steps'].append(episode_steps)
        
        # Collect full population history every 10 episodes for plotting
        if episode % 10 == 0:
            full_population_history['prey'].extend(env.population_history['prey'])
            full_population_history['predator'].extend(env.population_history['predator'])
            full_population_history['timesteps'].extend(
                [t + len(full_population_history['timesteps']) 
                 for t in env.population_history['timesteps']]
            )
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f"\nEpisode {episode}:")
            print(f"  Prey: {final_prey}, Predators: {final_predator}")
            print(f"  Avg Prey Reward: {stats['avg_prey_reward']:.2f}")
            print(f"  Avg Predator Reward: {stats['avg_predator_reward']:.2f}")
            print(f"  Epsilon: {stats['epsilon']:.3f}")
    
    # Save results
    print("\nSaving results...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Q-tables
    ql_manager.save_all(os.path.join(save_dir, 'q_tables'))
    
    # Save statistics
    with open(os.path.join(save_dir, 'training_stats.pkl'), 'wb') as f:
        pickle.dump(episode_stats, f)
    
    with open(os.path.join(save_dir, 'population_history.pkl'), 'wb') as f:
        pickle.dump(full_population_history, f)
    
    print("Training complete!")
    
    return env, ql_manager, episode_stats, full_population_history


def plot_results(episode_stats, population_history, save_dir='/mnt/user-data/outputs'):
    """
    Create visualizations of training results
    Includes Lotka-Volterra style population dynamics plot
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Population over episodes
    ax = axes[0, 0]
    ax.plot(episode_stats['episode'], episode_stats['prey_count'], 
            label='Prey', color='green', alpha=0.7)
    ax.plot(episode_stats['episode'], episode_stats['predator_count'], 
            label='Predator', color='red', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Population')
    ax.set_title('Population Dynamics Over Training Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Lotka-Volterra style continuous population plot
    ax = axes[0, 1]
    ax.plot(population_history['timesteps'], population_history['prey'], 
            label='Prey', color='green', linewidth=2)
    ax.plot(population_history['timesteps'], population_history['predator'], 
            label='Predator', color='red', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.set_title('Lotka-Volterra Model After Reinforcement Learning')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Rewards over training
    ax = axes[1, 0]
    # Smooth rewards with moving average
    window = 50
    if len(episode_stats['avg_prey_reward']) > window:
        prey_smooth = np.convolve(episode_stats['avg_prey_reward'], 
                                  np.ones(window)/window, mode='valid')
        predator_smooth = np.convolve(episode_stats['avg_predator_reward'], 
                                      np.ones(window)/window, mode='valid')
        episodes_smooth = episode_stats['episode'][window-1:]
        
        ax.plot(episodes_smooth, prey_smooth, label='Prey (smoothed)', color='green')
        ax.plot(episodes_smooth, predator_smooth, label='Predator (smoothed)', color='red')
    else:
        ax.plot(episode_stats['episode'], episode_stats['avg_prey_reward'], 
                label='Prey', color='green', alpha=0.5)
        ax.plot(episode_stats['episode'], episode_stats['avg_predator_reward'], 
                label='Predator', color='red', alpha=0.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Learning Progress: Average Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Exploration rate (epsilon) decay
    ax = axes[1, 1]
    ax.plot(episode_stats['episode'], episode_stats['epsilon'], 
            color='blue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon (Exploration Rate)')
    ax.set_title('Exploration vs Exploitation Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    print(f"Plots saved to {os.path.join(save_dir, 'training_results.png')}")
    
    return fig


def compare_with_lotka_volterra(population_history, save_dir='/mnt/user-data/outputs'):
    """
    Create theoretical Lotka-Volterra comparison
    """
    from scipy.integrate import odeint
    
    # Lotka-Volterra parameters (matching report constants)
    alpha = 0.15   # Prey growth rate
    beta = 0.01    # Predation rate
    gamma = 0.01   # Predator death rate
    delta = 0.005  # Predator growth from eating
    
    def lotka_volterra(state, t):
        x, y = state  # prey, predator
        dx = alpha * x - beta * x * y
        dy = delta * x * y - gamma * y
        return [dx, dy]
    
    # Initial conditions from RL data
    x0 = population_history['prey'][0] if population_history['prey'] else 5
    y0 = population_history['predator'][0] if population_history['predator'] else 5
    
    # Time points
    t = np.linspace(0, 1000, 1000)
    
    # Solve ODE
    solution = odeint(lotka_volterra, [x0, y0], t)
    
    # Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # RL results
    ax.plot(population_history['timesteps'][:1000], 
            population_history['prey'][:1000], 
            label='RL Prey', color='green', linewidth=2, alpha=0.7)
    ax.plot(population_history['timesteps'][:1000], 
            population_history['predator'][:1000], 
            label='RL Predator', color='red', linewidth=2, alpha=0.7)
    
    # Theoretical LV
    ax.plot(t, solution[:, 0], 
            label='LV Theory Prey', color='darkgreen', 
            linestyle='--', linewidth=2, alpha=0.7)
    ax.plot(t, solution[:, 1], 
            label='LV Theory Predator', color='darkred', 
            linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title('RL Results vs Lotka-Volterra Theoretical Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lotka_volterra_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {os.path.join(save_dir, 'lotka_volterra_comparison.png')}")
    
    return fig


if __name__ == "__main__":
    # Train the model
    env, ql_manager, stats, pop_history = train_predator_prey(
        num_episodes=500,
        num_prey=5,
        num_predator=5,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_decay=0.995
    )
    
    # Create visualizations
    plot_results(stats, pop_history)
    compare_with_lotka_volterra(pop_history)
    
    print("\n" + "=" * 60)
    print("Training and analysis complete!")
    print("Results saved to /mnt/user-data/outputs/")
    print("=" * 60)

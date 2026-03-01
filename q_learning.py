"""
Q-Learning Implementation for Multi-Agent Predator-Prey System

Implements:
- Q-table based learning for discrete state/action spaces
- Epsilon-greedy exploration
- Bellman equation updates
- Independent learners (each agent has its own Q-table)
"""

import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, Tuple, Optional
import random


class QLearningAgent:
    """
    Q-Learning agent for predator-prey environment
    
    Implements the Bellman equation:
    Q(s,a) = Q(s,a) + α[R + γ * max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        agent_type: str,  # 'prey' or 'predator'
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.agent_type = agent_type
        self.alpha = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: Q[state_hash][action] = value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Action space
        if agent_type == 'prey':
            self.num_actions = 7
        else:  # predator
            self.num_actions = 9
        
        # Statistics
        self.total_reward = 0
        self.episode_rewards = []
    
    def _discretize_state(self, observation: np.ndarray) -> str:
        """
        Convert continuous observation to discrete state
        
        For prey (36 dims): discretize energy + vision
        For predator (15 dims): discretize energy + vision + direction
        """
        # Discretize into bins
        discretized = []
        
        if self.agent_type == 'prey':
            # Energy: 0-100 → 5 bins
            energy_bin = int(observation[0] * 4)  # 0-4
            discretized.append(str(energy_bin))
            
            # Vision: simplify to counts
            prey_count = int(np.sum(observation[1:36] == 1.0))
            pred_count = int(np.sum(observation[1:36] == 2.0))
            
            # Bin counts
            prey_bin = min(prey_count, 5)  # 0-5
            pred_bin = min(pred_count, 5)  # 0-5
            
            discretized.append(str(prey_bin))
            discretized.append(str(pred_bin))
            
        else:  # predator
            # Energy
            energy_bin = int(observation[0] * 4)
            discretized.append(str(energy_bin))
            
            # Vision counts
            prey_count = int(np.sum(observation[1:10] == 1.0))
            prey_bin = min(prey_count, 3)
            discretized.append(str(prey_bin))
            
            # Direction
            direction = int(np.argmax(observation[10:16]))
            discretized.append(str(direction))
        
        return '_'.join(discretized)
    
    def get_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            observation: Current state observation
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            Selected action (0 to num_actions-1)
        """
        state = self._discretize_state(observation)
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: best action from Q-table
            q_values = [self.q_table[state][a] for a in range(self.num_actions)]
            max_q = max(q_values)
            
            # Handle ties randomly
            best_actions = [a for a in range(self.num_actions) if q_values[a] == max_q]
            return random.choice(best_actions)
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Update Q-value using Bellman equation
        
        Q(s,a) ← Q(s,a) + α[R + γ * max_a' Q(s',a') - Q(s,a)]
        """
        s = self._discretize_state(state)
        s_next = self._discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[s][action]
        
        # Next state max Q-value
        if done:
            max_next_q = 0.0
        else:
            next_q_values = [self.q_table[s_next][a] for a in range(self.num_actions)]
            max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Bellman update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[s][action] = new_q
        
        # Track reward
        self.total_reward += reward
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save Q-table to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load(self, filepath: str):
        """Load Q-table from file"""
        with open(filepath, 'rb') as f:
            loaded_table = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), loaded_table)


class MultiAgentQLearning:
    """
    Manages multiple Q-learning agents in the predator-prey environment
    
    Uses Independent Q-Learning: each agent learns independently
    """
    
    def __init__(
        self,
        prey_agents: list,
        predator_agents: list,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        # Create Q-learning agent for each entity
        self.agents = {}
        
        for agent in prey_agents:
            self.agents[agent] = QLearningAgent(
                agent_type='prey',
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay
            )
        
        for agent in predator_agents:
            self.agents[agent] = QLearningAgent(
                agent_type='predator',
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay
            )
    
    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        training: bool = True
    ) -> Dict[str, int]:
        """Get actions for all agents"""
        actions = {}
        for agent_name, obs in observations.items():
            if agent_name in self.agents:
                actions[agent_name] = self.agents[agent_name].get_action(obs, training)
        return actions
    
    def update(
        self,
        agent_name: str,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Update Q-values for a specific agent"""
        if agent_name in self.agents:
            self.agents[agent_name].update(state, action, reward, next_state, done)
    
    def decay_epsilon(self):
        """Decay exploration for all agents"""
        for agent in self.agents.values():
            agent.decay_epsilon()
    
    def get_statistics(self) -> Dict:
        """Get learning statistics"""
        prey_rewards = [
            agent.total_reward 
            for name, agent in self.agents.items() 
            if 'prey' in name
        ]
        predator_rewards = [
            agent.total_reward 
            for name, agent in self.agents.items() 
            if 'predator' in name
        ]
        
        return {
            'avg_prey_reward': np.mean(prey_rewards) if prey_rewards else 0,
            'avg_predator_reward': np.mean(predator_rewards) if predator_rewards else 0,
            'epsilon': self.agents[list(self.agents.keys())[0]].epsilon if self.agents else 0
        }
    
    def save_all(self, directory: str):
        """Save all Q-tables"""
        import os
        os.makedirs(directory, exist_ok=True)
        for agent_name, agent in self.agents.items():
            filepath = os.path.join(directory, f"{agent_name}_qtable.pkl")
            agent.save(filepath)
    
    def load_all(self, directory: str):
        """Load all Q-tables"""
        import os
        for agent_name, agent in self.agents.items():
            filepath = os.path.join(directory, f"{agent_name}_qtable.pkl")
            if os.path.exists(filepath):
                agent.load(filepath)

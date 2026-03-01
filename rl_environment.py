"""
Reinforcement Learning Environment for Predator-Prey Simulation
Standalone version - no external dependencies except standard libraries
"""

import functools
import random
from copy import copy, deepcopy
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from typing import Dict, List, Tuple, Optional

# Constants
NO_OF_PREY = 5
NO_OF_PREDATOR = 5
GRID_SIZE = (79, 39)
MAX_STEPS = 1000

# Energy parameters (matching report)
ALPHA = 15   # Prey energy gain while staying
DELTA = 50   # Predator energy gain while eating
GAMMA = 1    # Predator energy loss while moving


def list_to_axial(index: int) -> Tuple[int, int]:
    """Convert list index to axial coordinates"""
    quotient = int(index / 79)
    index = index % 79
    x = index
    y = -1 * int(index / 2) + quotient
    return (x, y)


def axial_to_list(axial: Tuple[int, int]) -> int:
    """Convert axial coordinates to list index"""
    index = axial[0] + (axial[1] + int(axial[0] / 2)) * 79
    index = index % (79 * 39)
    return index


def direction_generator(currentx: int, currenty: int) -> Tuple[int, int]:
    """Generate random neighboring position"""
    direction = random.randint(1, 6)
    if direction == 1:
        return (currentx, currenty - 1)
    elif direction == 2:
        return (currentx + 1, currenty - 1)
    elif direction == 3:
        return (currentx + 1, currenty)
    elif direction == 4:
        return (currentx, currenty + 1)
    elif direction == 5:
        return (currentx - 1, currenty + 1)
    elif direction == 6:
        return (currentx - 1, currenty)
    return (currentx, currenty)


def list_neighbours(axial: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get neighboring hexagons"""
    x, y = axial
    return [
        (x, y - 1),
        (x + 1, y - 1),
        (x + 1, y),
        (x, y + 1),
        (x - 1, y + 1),
        (x - 1, y)
    ]


def prey_vision(axial: Tuple[int, int], radius: int) -> List[Tuple[int, int]]:
    """Get all cells visible to prey (circular vision)"""
    nodes = {(axial[0], axial[1])}
    
    for i in range(radius + 1):
        if i == 1:
            neighbours = list_neighbours(axial)
            for neighbour in neighbours:
                nodes.add(neighbour)
        elif i > 1:
            for neighbour in list(nodes):
                temp_neighbours = list_neighbours(neighbour)
                for temp_neighbour in temp_neighbours:
                    nodes.add(temp_neighbour)
    
    nodes.discard((axial[0], axial[1]))
    return list(nodes)


def predator_vision(axial: Tuple[int, int], direction: int) -> List[Tuple[int, int]]:
    """Get cells visible to predator (directional cone)"""
    x, y = axial
    predVis = []
    
    # Direction vectors
    if direction == 1:
        a, b, c, d = 1, -1, -1, 0
    elif direction == 2:
        a, b, c, d = 1, 0, 0, -1
    elif direction == 3:
        a, b, c, d = 0, 1, 1, -1
    elif direction == 4:
        a, b, c, d = -1, 1, 1, 0
    elif direction == 5:
        a, b, c, d = -1, 0, 0, 1
    elif direction == 6:
        a, b, c, d = 0, -1, -1, 1
    else:
        return [(x, y)]
    
    # Build vision cone
    for i in range(3):
        for j in range(3):
            check_x = x + j * c + i * a
            check_y = y + j * d + i * b
            predVis.append((check_x, check_y))
    
    return predVis


class PredatorPreyEnv(ParallelEnv):
    """
    Multi-Agent Reinforcement Learning Environment for Predator-Prey Dynamics
    
    State Space:
        - Prey: 36-dimensional (energy + 35 vision cells with agent info)
        - Predator: 15-dimensional (energy + 9 vision cells + 5 directional info)
    
    Action Space:
        - Prey: 7 actions (6 directions + stay)
        - Predator: 9 actions (6 directions + stay + 2 rotation)
    """
    
    metadata = {
        "name": "predator_prey_rl_v0",
        "render_modes": ["human", "rgb_array", None],
    }
    
    def __init__(self, render_mode=None, num_prey=5, num_predator=5):
        self.num_prey = num_prey
        self.num_predator = num_predator
        
        # Agent lists
        self.possible_prey = [f"prey_{i}" for i in range(num_prey)]
        self.possible_predator = [f"predator_{i}" for i in range(num_predator)]
        self.possible_agents = self.possible_prey + self.possible_predator
        
        # State tracking
        self.prey_data = {}
        self.predator_data = {}
        
        self.agents = []
        self.timestep = 0
        self.render_mode = render_mode
        
        # Population tracking for Lotka-Volterra comparison
        self.population_history = {
            'prey': [],
            'predator': [],
            'timesteps': []
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.agents = self.possible_agents.copy()
        self.timestep = 0
        
        # Initialize prey agents
        self.prey_data = {}
        for agent in self.possible_prey:
            pos = list_to_axial(random.randint(0, 3080))
            self.prey_data[agent] = {
                'x': pos[0],
                'y': pos[1],
                'energy': random.randint(50, 100)
            }
        
        # Initialize predator agents
        self.predator_data = {}
        for agent in self.possible_predator:
            pos = list_to_axial(random.randint(0, 3080))
            self.predator_data[agent] = {
                'x': pos[0],
                'y': pos[1],
                'energy': random.randint(80, 100),
                'direction': random.randint(1, 6)
            }
        
        # Reset population tracking
        self.population_history = {
            'prey': [len(self.possible_prey)],
            'predator': [len(self.possible_predator)],
            'timesteps': [0]
        }
        
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def _get_observation(self, agent: str) -> np.ndarray:
        """Get observation for an agent"""
        
        if agent in self.prey_data:
            # Prey observation (36 dims)
            obs = np.zeros(36)
            data = self.prey_data[agent]
            
            # Energy (normalized)
            obs[0] = min(data['energy'] / 100.0, 1.0)
            
            # Vision
            vision_cells = prey_vision((data['x'], data['y']), 3)
            for i, cell in enumerate(vision_cells[:35]):
                cell_content = self._get_cell_content(cell)
                obs[i + 1] = cell_content
            
            return obs
        
        else:
            # Predator observation (15 dims)
            obs = np.zeros(15)
            data = self.predator_data[agent]
            
            # Energy (normalized)
            obs[0] = min(data['energy'] / 100.0, 1.0)
            
            # Directional vision
            vision_cells = predator_vision((data['x'], data['y']), data['direction'])
            for i, cell in enumerate(vision_cells[:9]):
                cell_content = self._get_cell_content(cell)
                obs[i + 1] = cell_content
            
            # Direction (one-hot) - directions 1-6 map to indices 9-14
            if data['direction'] >= 1 and data['direction'] <= 6:
                obs[9 + data['direction'] - 1] = 1.0
            
            return obs
    
    def _get_cell_content(self, pos: Tuple[int, int]) -> float:
        """Check what's in a cell: 0=empty, 1=prey, 2=predator"""
        # Check prey
        for agent, data in self.prey_data.items():
            if agent in self.agents and (data['x'], data['y']) == pos:
                return 1.0
        
        # Check predators
        for agent, data in self.predator_data.items():
            if agent in self.agents and (data['x'], data['y']) == pos:
                return 2.0
        
        return 0.0
    
    def step(self, actions: Dict[str, int]):
        """Execute actions for all agents"""
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        # Execute prey actions
        for agent in list(self.agents):
            if agent not in self.prey_data or agent not in actions:
                continue
            
            action = actions[agent]
            data = self.prey_data[agent]
            
            if action == 6:  # Stay action (0-indexed, so 6 is action 7)
                data['energy'] += ALPHA
                rewards[agent] = 1.0
            else:
                # Move
                new_x, new_y = self._apply_direction(data['x'], data['y'], action + 1)
                data['x'] = new_x
                data['y'] = new_y
                rewards[agent] = -0.1
        
        # Execute predator actions
        for agent in list(self.agents):
            if agent not in self.predator_data or agent not in actions:
                continue
            
            action = actions[agent]
            data = self.predator_data[agent]
            
            # Lose energy per timestep
            data['energy'] -= GAMMA
            
            if action < 6:  # Move actions
                new_x, new_y = self._apply_direction(data['x'], data['y'], action + 1)
                data['x'] = new_x
                data['y'] = new_y
            elif action == 6:  # Stay
                pass
            elif action == 7:  # Rotate left
                data['direction'] = (data['direction'] % 6) + 1
            elif action == 8:  # Rotate right
                data['direction'] = ((data['direction'] - 2) % 6) + 1
        
        # Check predation
        self._handle_predation(rewards, terminations)
        
        # Check energy death and reproduction
        self._handle_energy_mechanics(rewards, terminations)
        
        # Update agents list
        self.agents = [a for a in self.agents if not terminations.get(a, False)]
        
        # Check truncation
        self.timestep += 1
        if self.timestep >= MAX_STEPS:
            truncations = {agent: True for agent in self.agents}
        
        # Track population
        prey_count = sum(1 for a in self.agents if a in self.prey_data)
        predator_count = sum(1 for a in self.agents if a in self.predator_data)
        self.population_history['prey'].append(prey_count)
        self.population_history['predator'].append(predator_count)
        self.population_history['timesteps'].append(self.timestep)
        
        # Get new observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _apply_direction(self, x: int, y: int, direction: int) -> Tuple[int, int]:
        """Apply movement direction and handle wrapping"""
        if direction == 1:
            new_x, new_y = x, y - 1
        elif direction == 2:
            new_x, new_y = x + 1, y - 1
        elif direction == 3:
            new_x, new_y = x + 1, y
        elif direction == 4:
            new_x, new_y = x, y + 1
        elif direction == 5:
            new_x, new_y = x - 1, y + 1
        elif direction == 6:
            new_x, new_y = x - 1, y
        else:
            new_x, new_y = x, y
        
        # Wrap using axial/list conversion
        wrapped_pos = list_to_axial(axial_to_list((new_x, new_y)))
        return wrapped_pos
    
    def _handle_predation(self, rewards: Dict, terminations: Dict):
        """Handle predator eating prey"""
        for pred_agent, pred_data in self.predator_data.items():
            if pred_agent not in self.agents:
                continue
            
            pred_pos = (pred_data['x'], pred_data['y'])
            
            for prey_agent, prey_data in list(self.prey_data.items()):
                if prey_agent not in self.agents:
                    continue
                
                prey_pos = (prey_data['x'], prey_data['y'])
                
                if pred_pos == prey_pos:
                    pred_data['energy'] += DELTA
                    rewards[pred_agent] = 20.0
                    rewards[prey_agent] = -10.0
                    terminations[prey_agent] = True
                    break
    
    def _handle_energy_mechanics(self, rewards: Dict, terminations: Dict):
        """Handle death from low energy and reproduction from high energy"""
        
        # Check prey
        for agent, data in list(self.prey_data.items()):
            if agent not in self.agents:
                continue
            
            # Reproduction
            if data['energy'] >= 100:
                offspring_name = f"{agent}_child_{self.timestep}"
                new_pos = direction_generator(data['x'], data['y'])
                
                self.prey_data[offspring_name] = {
                    'x': new_pos[0],
                    'y': new_pos[1],
                    'energy': data['energy'] // 2
                }
                data['energy'] = data['energy'] // 2
                self.agents.append(offspring_name)
                rewards[agent] = 5.0
        
        # Check predators
        for agent, data in list(self.predator_data.items()):
            if agent not in self.agents:
                continue
            
            # Death from starvation
            if data['energy'] <= 0:
                terminations[agent] = True
                rewards[agent] = -20.0
                continue
            
            # Reproduction
            if data['energy'] >= 100:
                offspring_name = f"{agent}_child_{self.timestep}"
                new_pos = direction_generator(data['x'], data['y'])
                
                self.predator_data[offspring_name] = {
                    'x': new_pos[0],
                    'y': new_pos[1],
                    'energy': data['energy'] // 2,
                    'direction': data['direction']
                }
                data['energy'] = data['energy'] // 2
                self.agents.append(offspring_name)
                rewards[agent] = 5.0
    
    def render(self):
        """Render the environment (text-based)"""
        if self.render_mode is None:
            return
        
        prey_count = sum(1 for a in self.agents if a in self.prey_data)
        pred_count = sum(1 for a in self.agents if a in self.predator_data)
        
        print(f"Step {self.timestep}: Prey={prey_count}, Predators={pred_count}")
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Define observation space"""
        if "prey" in agent:
            return Box(low=0, high=2, shape=(36,), dtype=np.float32)
        else:
            return Box(low=0, high=2, shape=(15,), dtype=np.float32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Define action space"""
        if "prey" in agent:
            return Discrete(7)
        else:
            return Discrete(9)

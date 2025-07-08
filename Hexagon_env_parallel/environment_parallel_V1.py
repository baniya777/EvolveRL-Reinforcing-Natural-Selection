import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
import pygame
from pettingzoo import ParallelEnv


from hexagon_env_parallel.helperDisplay import clearGrid, init_hexagons, initializeAgent, render, renderAgents
from hexagon_env_parallel.helperDisplay import predatorDirectionGenerator,preyDirectionGenerator,predator_vision,prey_vision,direction_generator
from hexagon_env_parallel.helperDisplay import predatorBehaviourCheck

NO_OF_PREY = 5
NO_OF_PREDATOR = 5
GRID_SIZE = (79,39)
# prey energy gain while staying
ALPHA = 15   # natural growth
# predator energy gain while eating
DELTA = 50   # 
# predator energy loss while moving
GAMMA = 1
PREY_REWARD = 20


class Hex_Env(ParallelEnv):

    metadata = {
        "name": "hex_env_parallel_v0",
    }

    def __init__(self, render_mode=None):

        
        
        self.possible_prey = ["prey_" + str(r) for r in range(NO_OF_PREY)]
        self.possible_predator = ["predator_" + str(r) for r in range(NO_OF_PREDATOR)]
        self.possible_agents = self.possible_prey.copy() + self.possible_predator.copy()

        self.prey_agents = self.possible_prey.copy()
        self.predator_agents = self.possible_predator.copy()

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = dict([(agent, Discrete(7)) for agent in self.possible_prey] + [(agent,Discrete(9)) for agent in self.possible_predator])
        self._observation_spaces = dict([(agent, Discrete(36)) for agent in self.possible_prey] + [(agent,Discrete(15)) for agent in self.possible_predator])
        self.predatorAgents,self.preyAgents = initializeAgent(NO_OF_PREDATOR,NO_OF_PREY)

        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            self.clock = pygame.time.Clock()
            self.world = pygame.display.set_mode((1200, 700))
            self.hexagons = init_hexagons(flat_top=True)
            self.clock.tick(70)


    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}

        self.observations = dict([(agent,[0]*36)  for agent in self.prey_agents]+[(agent,[0]*15)  for agent in self.predator_agents])
        self.num_moves = 0
        self.prey_agents = self.possible_prey.copy()
        self.predator_agents = self.possible_predator.copy()

        return self.observations, self.infos
    

    def step(self, actions):
        # Check termination conditions
        terminations = dict([(agent, False) for agent in self.possible_prey] + [(agent,False) for agent in self.possible_predator])
        truncations = dict([(agent, False) for agent in self.possible_prey] + [(agent,False) for agent in self.possible_predator])
        rewards = dict([(agent, 0) for agent in self.possible_prey] + [(agent,0) for agent in self.possible_predator])
        
        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        
        # to take actions for all agents
        for i in actions.keys():
            if i[3] == 'd':    # implies that i is predator
                index = self.predator_agents.index(i)
                x,y,dir =predatorDirectionGenerator(self.predatorAgents[index][0],self.predatorAgents[index][1],self.predatorAgents[index][4],actions[i])
                self.predatorAgents[index][0] = x
                self.predatorAgents[index][1] = y           
                self.predatorAgents[index][4] = dir
            else:
                index = self.prey_agents.index(i)
                # if stayed at same place increase energy
                if actions[i] == 7 :
                    self.prey_agents[index][3] += 15
                    self.rewards[i] = PREY_REWARD
                else:
                    # elif moved from a place then move to that place
                    x,y = preyDirectionGenerator(self.preyAgents[index][0],self.preyAgents[index][1],actions[i])
                    self.preyAgents[index][0] = x
                    self.preyAgents[index][1] = y
                
        # check behaviour after actions have been taken
        # change energy level for predator 
        predatorBehaviourCheck(self.predator_agents,self.predator_agents,self.rewards,self.terminations)
        # change energy level for prey if completed the stay at a place

        self.timestep += 1
        
        # Get observations

        # Set up observations
        observations = {
            a: 0
            for a in self.agents
        }

        infos = {a: {} for a in self.agents}

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            print('no agent specified')
            return
        
        if len(self.prey_agents) > 0 and len(self.predator_agents) > 0:
            clearGrid(self.hexagons)
            renderAgents(self.preyAgents,self.predatorAgents,self.hexagons)
            render(self.world, self.hexagons)

        else:
            string = "Game over"
            print(string)
            self.close()

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pygame.display.quit()



    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if (agent[3] == 'd'):
            return Discrete(15)
        else:
            return Discrete(36)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if (agent[3] =='d'):
            return Discrete(9)
        else:
            return Discrete(7)

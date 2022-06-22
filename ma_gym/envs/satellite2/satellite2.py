import copy
import logging
import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding
import random
from itertools import cycle, islice 
from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.replay_buffer import ReplayBuffer

"""
Satellite environment involve a grid world, in which multiple satellite attempt to observe different rocks or region of rocks.The goal is to slew the main satellite (mothership) towards the direction of the rocks so they are in the observable range for data collection and maximize data collection while maintaining battery and memory levels. In this environment, the satellite is constantly going up and will reset itself once it hits the top and moves over 3 columns until it hits [0,19] in which the episode ends. 

LIMITATIONS OF AGENT/S:
- Agent/s is not allowed to move more than three units to the left or right before the reset happens
- Agent/s must charge once the battery is depleted which takes three time steps
- Agent/s must also dump memory every time the memory level is equal to 3


Agent/s can select one of the actions in the action_space ∈ {Up (No-op), Left, Right}. The agent is always moving up to represent "floating".

Each agent's observation includes its:
    - Agent ID 
    - Position within the grid
    - Number of steps since beginning
    - Battery Level |Partially empty = 3 (Takes 1 step), Half empty = 2 (Takes 2 steps), Empty = 1 (Takes 3 steps)| 
    - Memory Level |Partially full  = 1(Takes 1 step), Half full = 2 (Takes 2 steps), Full = 3 (Takes 3 steps)|
    - Observed Status |Partially Observed = 1 (Takes 1 step), Half Observed = 2 (Takes 2 steps), 
                       Fully Observed = 3 (Takes 3 steps)|

-----------------------
HOW TO DO TELE-OP 
_______________________
Run interactive_agent.py
Control Using:
0: NO-OP
1: LEFT 
2: UP
3: RIGHT
-----------------------

Only the agents who are involved in observation recieve a reward.
The environment is terminated as soon as goals for satellite/s are reached. 
Upon rendering with render(), we show the grid, where each cell shows the agents (blue) and rocks (orange).

Arguments:
    grid_shape: size of the grid
    n_agents: number of agents/satellites
    n_rocks: number of rocks
    agent_view: size of the agent view range in each direction
    full_observable: flag whether agents should receive observation for all other agents

Attributes:
    _agent_dones: list with indicater whether the agent is done or not.
    _base_img: base image with grid
    _viewer: viewer for the rendered image
    _num_rocks_found : total number of rocks/regions of rocks found after each time step

"""


class Satellites2(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']} #must be human so it is readable data by humans

    def __init__(self, grid_shape=(20, 20), n_agents=1, n_rocks=10, full_observable=False, max_steps=10, 
                 agent_view_mask=(20, 20)):

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_rocks = n_rocks
        self._max_steps = max_steps
        self._step_count = None
        self._agent_view_mask = agent_view_mask
        self._init_agent_pos = {_: None for _ in range(self.n_agents)}

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.rocks_pos = {_: None for _ in range(self.n_rocks)}

        self.agent_battery = {_: None for _ in range(self.n_agents)}
        self.agent_memory =  {_: None for _ in range(self.n_agents)}
        self.agent_observed =  {_: None for _ in range(self.n_agents)}

        self._num_rocks_found = 0

        self._base_grid = self.__create_grid() 
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
 
        self.viewer = None
        self.full_observable = full_observable


        mask_size = np.prod(self._agent_view_mask)
        self._obs_high = np.array([1., 1.] + [1.] * mask_size + [1.0], dtype=np.float32)
        self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0], dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()
    """
        Function: get_action_meanings()
        Inputs  : agent_i 
        Outputs : None
        Purpose : Reports back what move agent/user chose
    """
    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def sample_action_space(self): # only for RL  
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self): # draws empty grid
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __init_map(self):
        self._full_obs = self.__create_grid()

        for agent_i in range(self.n_agents):
            self.agent_memory = [0.0]
            self.agent_battery = [3.0]
            while True:
                # pos = [self.np_random.randint(0, self._grid_shape[0] - 1), #for multiple agents
                # self.np_random.randint(0, self._grid_shape[1] - 1)]
                pos = [self._grid_shape[0] - 4,1] # for one agent
                if self._is_cell_vacant(pos):
                    self.agent_pos[agent_i] = pos
                    self._init_agent_pos = pos
                    break
            self.__update_agent_view(agent_i)

        for rock_i in range(self.n_rocks):
            while True:
                pos = [self.np_random.randint(2, self._grid_shape[0] - 1), #leave space for satellite to respawn
                       self.np_random.randint(2, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (self._neighbor_agents(pos)[0] == 0 and pos[1] % 3 == 0):#dont spawn same place as agent
                    self.rock_pos[rock_i] = pos
                    break
            self.__update_rock_view(rock_i)

        self.__draw_base_img()
    """
        Function : get_agent_obs()
        Inputs   : None
        Outputs  : full observation of the grid world
        Purpose  : 
    """
    def get_agent_obs(self):
        print("Getting Agent Obs")
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            print("Agent Pos: {}".format(pos))
            battery_level = self.agent_battery[agent_i]
            memory_level = self.agent_memory[agent_i]

            _agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  #coordinate of agent

            # check if rock is in the view area and give it future (time+1) rock coordinates
            _rock_pos = np.zeros(self._agent_view_mask)  # rock location in neighbor
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['rock'] in self._full_obs[row][col]:
                        _rock_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the rock loc.
                        # print("Observation space \n {}".format(_rock_pos))
            _agent_i_obs += _rock_pos.flatten().tolist()  # adding rock pos in observable area
            _agent_i_obs += [self._step_count / self._max_steps]  # adding the time

            _agent_i_obs += [battery_level]
            _agent_i_obs += [memory_level]

            _obs.append(_agent_i_obs)

            # print("Obs {}".format(_obs))


        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist() # flatten to np array so it is able to be processed by NN
            _obs = [_obs for _ in range(self.n_agents)]
        
        return _obs

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.rock_pos = {}

        self.__init_map()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]

        return self.get_agent_obs()

    def __wall_exists(self, pos): #was in example env so ported over in case
        row, col = pos
        return PRE_IDS['wall'] in self._base_grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __next_pos(self, curr_pos, move):

        moves = {
            0: [curr_pos[0] - 1, curr_pos[1]],     # no-op
            1: [curr_pos[0] - 1, curr_pos[1] - 1], # left
            2: [curr_pos[0] - 1, curr_pos[1] + 1], # right
        }

        next_pos = moves[move]

        return next_pos

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None

        moves = {
            0: [curr_pos[0] - 1, curr_pos[1]],     # no-op
            1: [curr_pos[0] - 1, curr_pos[1] - 1], # left
            2: [curr_pos[0] - 1, curr_pos[1] + 1], # right
        }

        next_pos = moves[move]

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_rock_view(self, rock_i):
        self._full_obs[self.rock_pos[rock_i][0]][self.rock_pos[rock_i][1]] = PRE_IDS['rock'] + str(rock_i + 1)
    """
        Function : _neighbor_agents
        Inputs   : pos
        Outputs  : number of neighbors
        Purpose  : This function is the visual of the "observation space" for each agent. This is done by putting all the squares surrounding the agent as a "neighbor" and acts like an extension of the agent. 
    """
    def _neighbor_agents(self, pos): # this is really just the observable space 
        
        _count = 0
        neighbors_xy = []

        #Covers North, South, East, West
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbors_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbors_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbors_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbors_xy.append([pos[0], pos[1] - 1])
            _count += 1

        #Covers NE, SE, NW, SW
        if self.is_valid([pos[0] + 1, pos[1] +1 ]) and PRE_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1] + 1]:
            _count += 1
            neighbors_xy.append([pos[0] + 1, pos[1] + 1])
        if self.is_valid([pos[0] - 1, pos[1] - 1 ]) and PRE_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1] - 1]:
            _count += 1
            neighbors_xy.append([pos[0] - 1, pos[1] - 1])
        if self.is_valid([pos[0] - 1, pos[1] + 1]) and PRE_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1] + 1]:
            _count += 1
            neighbors_xy.append([pos[0] - 1, pos[1] + 1])
        if self.is_valid([pos[0] + 1, pos[1] - 1]) and PRE_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1] - 1]:
            neighbors_xy.append([pos[0] + 1, pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbors_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['agent'])[1]) - 1)
        return _count, agent_id

    def step(self, agents_action):
        self._step_count += 1
        rewards = [0 for _ in range(self.n_agents)]

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                print("agent_i is: {} and action is: {}".format(agent_i, action))
                self.__query_reset_path()
                self.__update_agent_pos(agent_i, action)
     
        return self.get_agent_obs(), rewards, self._agent_dones, {'filler': self._num_rocks_found}

    def __query_reset_path(self): #respawn agent once it hits bottom
        for agent_i in range(self.n_agents):
        
            init_pos = copy.copy(self._init_agent_pos)
            next_pos = None
            print("agent pos: ",self.agent_pos[0][0])
            print("grid shape: ", self._grid_shape[1])
            if(self.agent_pos[0][0] == 0):
                next_pos = [init_pos[0], init_pos[1] + 3]     
                print("reset {}".format(next_pos))
                if next_pos is not None and self._is_cell_vacant(next_pos):
                    self.agent_pos[agent_i] = next_pos
                    self._full_obs[init_pos[0]][init_pos[1]] = PRE_IDS['empty']
                    self.__update_agent_view(agent_i)
                self._init_agent_pos = next_pos

    def __get_neighbor_coordinates(self, pos): #figures out the pos of each block of obs space

        #Covers North, South, East, West
        neighbors = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbors.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbors.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbors.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbors.append([pos[0], pos[1] - 1])

        #Covers NE, SE, NW, SW
        if self.is_valid([pos[0] + 1, pos[1] +1 ]):
            neighbors.append([pos[0] + 1, pos[1] +1 ])
        if self.is_valid([pos[0] - 1, pos[1] - 1]):
            neighbors.append([pos[0] - 1, pos[1] - 1])
        if self.is_valid([pos[0] - 1, pos[1] + 1]):
            neighbors.append([pos[0] - 1, pos[1] + 1 ])
        if self.is_valid([pos[0] + 1, pos[1] - 1]):
            neighbors.append([pos[0] + 1, pos[1] - 1])
     
        return neighbors

    def render(self, mode='human'): #renders the entire grid world as one frame
        img = copy.copy(self._base_img)

        #visual render for the observable area of the satellite
        for agent_i in range(self.n_agents):
            for neighbor in self.__get_neighbor_coordinates(self.agent_pos[agent_i]):
                fill_cell(img, neighbor, cell_size=CELL_SIZE, fill=OBSERVATION_VISUAL_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=OBSERVATION_VISUAL_COLOR, margin=0.1)

        #agent visual
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)
        #rock visual
        for rock_i in range(self.n_rocks):
            draw_circle(img, self.rock_pos[rock_i], cell_size=CELL_SIZE, fill=ROCK_COLOR)
            write_cell_text(img, text=str(rock_i + 1), pos=self.rock_pos[rock_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


logger = logging.getLogger(__name__)

AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
OBSERVATION_VISUAL_COLOR = (186, 238, 247)
ROCK_COLOR = 'orange'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "NOOP",
    1: "LEFT",
    2: "RIGHT",
}

PRE_IDS = {

    'agent': 'A',
    'rock': 'P',
    'wall': 'W',
    'empty': '0'
}

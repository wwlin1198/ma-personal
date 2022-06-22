import copy
import itertools
import logging
from typing import List, Tuple, Union

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_circle, draw_grid, fill_cell, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
ROCK_COLOR = 'orange'
WALL_COLOR = 'black'

CELL_SIZE = 35

ACTIONS_IDS = {

    'none': 0,
    'down': 1,
    'left': 2,
    'up': 3,
    'right': 4,
}

PRE_IDS = {

    'empty': 0,
    'wall': 1,
    'agent': 2,
    'rock': 3,
}


Coordinates = Tuple[int, int]


"""
    Dataclass keeping all data for one agent/satellite in environment.

    Attributes:
        id: unique id in one environment run
        pos: position of the agent in grid
"""
class Agent:

    def __init__(self, id: int, pos: Coordinates):
        self.id = id
        self.pos = pos


"""
Satellite environment involve a grid world, in which multiple satellite attempt to observe different regions

Agents select one of fire actions ∈ {Up, Down, Left, Right}.
Each agent's observation includes its:
    - agent ID (1)
    - position within the grid (2)
    - number of steps since beginning (1)

All values are scaled down into range ∈ [0, 1].

Only the agents who are involved in observation recieve a reward, reward.
The environment is terminated as soon as goals for satellite/s are reached. 

Upon rendering, we show the grid, where each cell shows the agents (blue) and rocks (orange).

Args:
    grid_shape: size of the grid
    n_agents: number of agents/satellites
    n_rocks: number of rocks
    agent_view: size of the agent view range in each direction
    full_observable: flag whether agents should receive observation for all other agents
    step_cost: reward receive in each time step
    obs_reward: reward received by agents who completed observation of region
    max_steps: maximum steps in one environment episode

Attributes:
    _agents: list of all agents. The index in this list is also the ID of the agent
    _agent_map: three dimensional numpy array of indicators where the agents are located
    _rock_map: two dimensional numpy array of number of the roks
    _total_episode_reward: array with accumulated rewards for each agent.
    _agent_dones: list with indicater whether the agent is done or not.
    _base_img: base image with grid
    _viewer: viewer for the rendered image
"""
class Satellites(gym.Env):

    """
        Render Mode Human just means that when rendering the scenario, it will be able to be read by humans.
        Else, it can be ansi which is machine specific like \nSFFF\n\x1b[41mF\x1b[0mHFH.


        NOTE: number of rocks (n_rocks) recommended to be <100 for lower powered machines.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape: Coordinates = (20, 20), n_agents: int = 2, n_rocks: int = 30,
                 agent_view: Tuple[int, int] = (1, 1), full_observable: bool = False,
                 step_cost: float = -1, obs_reward: float = 10, max_steps: int = 100):
        assert 0 < n_agents
        assert n_agents + n_rocks <= np.prod(grid_shape)
        assert 1 <= agent_view[0] <= grid_shape[0] and 1 <= agent_view[1] <= grid_shape[1]

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._n_rocks = n_rocks
        self._agent_view = agent_view
        self.full_observable = full_observable
        self._step_cost = step_cost
        self._obs_reward = obs_reward
        self._max_steps = max_steps
        self.steps_beyond_done = 0
        self.seed()

        self._agents = []  # List[Agent]
        self._agent_map = None
        self._rock_map = None
        self._total_episode_reward = None
        self._agent_dones = None


        mask_size = np.prod(tuple(2 * v + 1 for v in self._agent_view))
        # Agent ID (1) + Pos (2) + Step (1) + Neighborhood (2 * mask_size)
        self._obs_len = (1 + 2 + 1 + 2 * mask_size)
        obs_high = np.array([1.] * self._obs_len, dtype=np.float32)
        obs_low = np.array([0.] * self._obs_len, dtype=np.float32)

        #this allows the entire grid to be observed which makes it easier for feature extraction
        if self.full_observable:
            obs_high = np.tile(obs_high, self.n_agents)
            obs_low = np.tile(obs_low, self.n_agents)
        self.action_space = MultiAgentActionSpace([spaces.Discrete(5)] * self.n_agents)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(obs_low, obs_high)] * self.n_agents)

        
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        self._viewer = None

    def _to_extended_coordinates(self, relative_coordinates):
        """Translate relative coordinates into the extended coordinates."""
        return relative_coordinates[0] + self._agent_view[0], relative_coordinates[1] + self._agent_view[1]

    def _to_relative_coordinates(self, extended_coordinates):
        """Translate extended coordinates into the relative coordinates."""
        return extended_coordinates[0] - self._agent_view[0], extended_coordinates[1] - self._agent_view[1]

    def _init_episode(self):
        """
        Initialize environment for new episode.

        """
        init_positions = self._generate_init_pos()
        agent_id, rock_id = 0, self.n_agents
        self._agents = []
        self._agent_map = np.zeros((
            self._grid_shape[0] + 2 * (self._agent_view[0]),
            self._grid_shape[1] + 2 * (self._agent_view[1]),
            self.n_agents
        ), dtype=np.int32)
        self._rock_map = np.zeros((
            self._grid_shape[0] + 2 * (self._agent_view[0]),
            self._grid_shape[1] + 2 * (self._agent_view[1]),
        ), dtype=np.int32)

        for pos, cell in np.ndenumerate(init_positions):
            pos = self._to_extended_coordinates(pos)
            if cell == PRE_IDS['agent']:
                self._agent_map[pos[0], pos[1], agent_id] = 1
                self._agents.append(Agent(agent_id, pos=pos))
                agent_id += 1
            elif cell == PRE_IDS['rock']:
                self._rock_map[pos] = self.np_random.randint(1, self.n_agents + 1)
                rock_id += 1


    #the -> defines the return value of this function
    def _generate_init_pos(self) -> np.ndarray:
        """Returns randomly selected initial positions for agents and rocks in relative coordinates.

        No agent or rocks share the same cell in initial positions.
        """
        init_pos = np.array(
            [PRE_IDS['agent']] * self.n_agents +
            [PRE_IDS['rock']] * self._n_rocks +
            [PRE_IDS['empty']] * (np.prod(self._grid_shape) - self.n_agents - self._n_rocks)
        )
        self.np_random.shuffle(init_pos)

        return np.reshape(init_pos, self._grid_shape)


    def step(self, agents_action: List[int]):

        return self.get_agent_obs()

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        mask = (
            slice(self._agent_view[0], self._agent_view[0] + self._grid_shape[0]),
            slice(self._agent_view[1], self._agent_view[1] + self._grid_shape[1]),
        )

        # Iterate over all grid positions
        for pos, agent_strength, rock_strength in self._view_generator(mask):
            if rock_strength and agent_strength:
                cell_size = (CELL_SIZE, CELL_SIZE / 2)
                rock_pos = (pos[0], 2 * pos[1])
                agent_pos = (pos[0], 2 * pos[1] + 1)
            else:
                cell_size = (CELL_SIZE, CELL_SIZE)
                rock_pos = agent_pos = (pos[0], pos[1])

            if rock_strength != 0:
                fill_cell(img, pos=rock_pos, cell_size=cell_size, fill=ROCK_COLOR, margin=0.1)
                write_cell_text(img, text=str(rock_strength), pos=rock_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

            if agent_strength != 0:
                draw_circle(img, pos=agent_pos, cell_size=cell_size, fill=AGENT_COLOR, radius=0.30)
                write_cell_text(img, text=str(agent_strength), pos=agent_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img)
            return self._viewer.isopen
            
    def _view_generator(self, mask: Tuple[slice, slice]) -> Tuple[Coordinates, int, int]:
        """
        Yields position, number of agent and rock strength for all cells defined by `mask`.

        Args:
            mask: tuple of slices in extended coordinates.
        """
        agent_iter = np.ndenumerate(np.sum(self._agent_map[mask], axis=2))
        rock_iter = np.nditer(self._rock_map[mask])
        for (pos, n_a), n_t in zip(agent_iter, rock_iter):
            yield pos, n_a, n_t 


    def _agent_generator(self) -> Tuple[int, Agent]:
        """Yields agent_id and agent for all agents in environment."""
        for agent_id, agent in enumerate(self._agents):
            yield agent_id, agent

    def seed(self, n: Union[None, int] = None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def reset(self) -> List[List[float]]:
        self._init_episode()
        self._step_count = 0
        self._total_episode_reward = np.zeros(self.n_agents)
        self._agent_dones = [False] * self.n_agents
        self.steps_beyond_done = 0

        return
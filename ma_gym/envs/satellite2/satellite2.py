import copy
import logging
from pickle import LONG4
import gym
import os
import torch as T
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding
from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.replay_buffer import ReplayBuffer
from ..utils.deep_q_network import DeepQNetwork

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

------------------------------------------------------------------------------------------------------------------------------
HOW TO DO TELE-OP 
_______________________
Run interactive_agent.py
Control Using:
0: NO-OP
1: LEFT 
2: UP
3: RIGHT
------------------------------------------------------------------------------------------------------------------------------

ACTION SPACE:
------------------------------------------------------------------------------------------------------------------------------

{observe, observe and move left, observe and move right, move left, move right, dump, charge }

                                                    Shortened to:
{O, OL, OR, L, R, D, C}

------------------------------------------------------------------------------------------------------------------------------
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

    def __init__(self, grid_shape=(20, 20), n_agents=1, n_rocks=10, full_observable=False, max_steps=100, 
                 agent_view_mask=(20, 20)):
        ############# PARAMS #################
        self.alpha          = 0.001 
        self.beta           = 0.002
        self.gamma          = 0.99
        self.epsilon        = 1
        self.eps_min        = 0.01
        self.eps_dec        = 1e-5
        self.replace        = 1000
        self.lr             = 0.001
        self.input_dims     = [405]
        self.n_actions      = 7
        self.mem_size       = 10000
        self.batch_size     = 512
        self.mem_cntr       = 0
        self.replace        = 1000
        self.iter_cntr      = 0
        self.learn_step_counter = 0
        self.replace_target_cnt = self.replace

        self.env_name       = "satellite2"
        self.algo           = ""
        self.checkpoint_dir = 'examples/checkpoints'

        ############# PARAMS #################

        ###########  For DQN ############
        self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)


        self.q_eval = DeepQNetwork(self.lr, n_actions=self.n_actions,
                                   input_dims=self.input_dims,
                                   fc1_dims=512, fc2_dims=512)
        ###########  For DQN ############
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_rocks = n_rocks
        self._max_steps = max_steps
        self._step_count = None
        self._agent_view_mask = agent_view_mask
        self._init_agent_pos = {_: None for _ in range(self.n_agents)}

        self.action_space = MultiAgentActionSpace([spaces.Discrete(7) for _ in range(self.n_agents)])
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
            self.agent_observed = [0.0]
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
            _obs = np.array(_obs).flatten().tolist() # flatten to np array 
            _obs = [_obs for _ in range(self.n_agents)]
        # _obs = tuple(map(tuple, _obs)) #convert to tuple
        
        return _obs

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.rock_pos = {}

        self.__init_map()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]

        return self.get_agent_obs()

    # def __wall_exists(self, pos): #was in example env so ported over in case
    #     row, col = pos
    #     return PRE_IDS['wall'] in self._base_grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    # def __next_pos(self, curr_pos, move):

    #     moves = {
    #         0: [curr_pos[0] - 1, curr_pos[1]],     # Observe
    #         1: [curr_pos[0] - 1, curr_pos[1] - 1], # Observe left
    #         2: [curr_pos[0] - 1, curr_pos[1] + 1], # Observe right

       
    #         3: [curr_pos[0] - 1, curr_pos[1] - 1], # move left
    #         4: [curr_pos[0] - 1, curr_pos[1] + 1], # move right

    #         5: [curr_pos[0] - 1, curr_pos[1]],     # Charge
    #         6: [curr_pos[0] - 1, curr_pos[1]],     # Dump no-op
    #     }

    #     next_pos = moves[move]

    #     return next_pos

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None

        moves = {
            0: [curr_pos[0] - 1, curr_pos[1]],     # Observe
            1: [curr_pos[0] - 1, curr_pos[1] - 1], # Observe left
            2: [curr_pos[0] - 1, curr_pos[1] + 1], # Observe right

       
            3: [curr_pos[0] - 1, curr_pos[1] - 1], # move left
            4: [curr_pos[0] - 1, curr_pos[1] + 1], # move right

            5: [curr_pos[0] - 1, curr_pos[1]],     # Charge
            6: [curr_pos[0] - 1, curr_pos[1]],     # Dump no-op
        }

        next_pos = moves[move]


        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self.__update_resources(agent_i,move)
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)


    def __update_resources(self,agent_i,action):
        if(action == OBSERVE_ID or action == OBSERVE_LEFT_ID or action == OBSERVE_RIGHT_ID):
            self.agent_memory[agent_i] += 1
            self.agent_observed[agent_i] += 1
            self.agent_battery[agent_i] -= 1
        if(action == LEFT_ID or action == RIGHT_ID):
            self.agent_battery[agent_i] -= 1
        if(action == DUMP_MEM_ID):
            self.agent_battery[agent_i] = 3
        if(action == CHARGE_ID):
            self.agent_memory[agent_i] = 0


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
        print("agents_action", agents_action)
        self._step_count += 1
        rewards = [0 for _ in range(self.n_agents)]
        in_range = 0
        _reward = 0
        for rock_i in range(self.n_rocks):# find if any rocks are in data
            rock_neighbour_count, n_i = self._neighbor_agents(self.rock_pos[rock_i])
            if rock_neighbour_count == 1: in_range += 1
        
        if(in_range <= 0 and agents_action[0] <= 3):#can't choose to observe if there are no rocks in range end ep.
            print("Invalid Move...")
            _reward = -10
            for agent_i in range(self.n_agents):
                self.agent_memory[agent_i] += 1
                rewards[agent_i] += _reward 

        if(in_range >= 1 and agents_action[0] > 3): #reward -10 if it is in collection range but decides not to
            _reward = -10
            for agent_i in range(self.n_agents):
                rewards[agent_i] += _reward 
        
        if(in_range >= 1 and agents_action[0] <= 3): #reward of 100 if they choose to observe when in data collection range
            _reward = 100
            for agent_i in range(self.n_agents):
                rewards[agent_i] += _reward 

        for agent_i in range(self.n_agents): #reward -10 if it doesn't choose to charge at 0 battery
            if(self.agent_battery[agent_i] <= 0 and agents_action[0] != 6): 
                _reward = -10
            if(self.agent_memory[agent_i] >= 3 and agents_action[0] != 6):
                _reward = -10
            if(self.agent_battery[agent_i] <= 0 and agents_action[0] != 5): #reward -10 if battery/memory is depleted/full
                _reward = -10
            rewards[agent_i] += _reward 

        
        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                print("agent_i is: {} and action is: {}".format(agent_i, action))
                print("Resource Levels: MEMORY LEVEL: {}, OBSERVED LEVEL: {}, BATTERY LEVEL: {}".\
                format(self.agent_memory[agent_i],self.agent_observed[agent_i],self.agent_battery[agent_i]))
                self.__query_reset_path()
                self.__update_agent_pos(agent_i, action)
  
        if (self._step_count >= self._max_steps):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]


        # print("Rewards: ",rewards)

        return self.get_agent_obs(), rewards, self._agent_dones, {'filler': self._num_rocks_found}

    def __query_reset_path(self): #respawn agent once it hits bottom
        for agent_i in range(self.n_agents):
            init_pos = copy.copy(self._init_agent_pos)
            next_pos = None
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
            
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation, evaluate=False): #e-greedy action selection
        # for agent_i in range (self.n_agents):
        #     if np.random.random() > self.epsilon:
        #         state_i = T.tensor(observation[agent_i],dtype=T.float).to(self.q_eval.device)
        #         actions = self.q_eval.forward(state_i)
        #         action = T.argmax(actions).item()
        #     else:
        #         action = self.sample_action_space()
        #     print("Action",action)
        #     return action
        # return None
        if np.random.random() > self.epsilon:
            print("randomsthistetjewoi")
            state_i = T.tensor(observation[0],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state_i)
            action = T.argmax(actions).item()
            action = [action]
        else:
            action = self.sample_action_space()
        print("Action",action)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        state, action, reward, new_state, done, batch = \
                                self.memory.sample_buffer(self.batch_size)
        
        # using T.Tensor seems to reset datatype to float
        # using T.tensor preserves source data type
        state = T.tensor(state).to(self.q_eval.device)
        new_state = T.tensor(new_state).to(self.q_eval.device)
        action = action
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        # state = T.squeeze(state)
        # new_state = T.squeeze(new_state)

        # print("state shape",state.shape)
        # print("new state shape",new_state.shape)
        # print("action shape",action)
        q_eval = self.q_eval.forward(state)[batch,action]
        q_next = self.q_eval.forward(new_state)
        q_next[dones] = 0
        # print("q_eval",q_eval)
        # print("q_next",q_next.shape)
  

        q_target = rewards + self.gamma*T.max(q_next, dim=1)[0]
        # print("q_target",q_target)


        loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
        # print("loss dtype",loss.dtype)
        # print("q_eval dtype", q_eval.dtype)
        # print("q_next dtype", q_next.dtype)
        # print("q_target dtype", q_target.dtype)
        # print("loss",loss)
        loss.backward()
        
        self.q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min


logger = logging.getLogger(__name__)

AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
OBSERVATION_VISUAL_COLOR = (186, 238, 247)
ROCK_COLOR = 'orange'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "O",
    1: "OL",
    2: "OR",

    3: "L",
    4: "R",

    5: "D",
    6: "C",

}

PRE_IDS = {
    'agent': 'A',
    'rock': 'P',
    'wall': 'W',
    'empty': '0'
}



OBSERVE_ID = 0
OBSERVE_LEFT_ID = 1
OBSERVE_RIGHT_ID = 2 
LEFT_ID = 3
RIGHT_ID = 4
DUMP_MEM_ID = 5
CHARGE_ID = 6
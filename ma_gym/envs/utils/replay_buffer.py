import tensorflow as tf
import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory,self.new_state_memory = tf.zeros((self.mem_size, *input_shape))
        self.action_memory = tf.zeros((self.mem_size, n_actions))
        self.reward_memory = tf.zeros(self.mem_size)
        self.terminal_memory = tf.zeros(self.mem_size, dtype=tf.float32)

    def store_transition(self, state, action, reward, state_, done): 
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
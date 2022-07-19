import argparse

import gym
import numpy as np

from ma_gym.wrappers import Monitor

from ma_gym.envs.utils.plot import plotLearning

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random env for ma-gym')
    parser.add_argument('--env', default='Satellite2-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()
    
    env = gym.make(args.env)
    env = Monitor(env, directory='recordings/' + args.env, force=True)
    n_games = 2000
    figure_file = 'recordings/satellite_learning.png'
    observation_file = 'recordings/satellite2_obs.png'


    best_score = -np.inf
    score_history = []
    obs_score_history = []
    load_checkpoint = False
    eps_history = []
    if load_checkpoint:
        n_steps = 0
        while n_steps <= env.batch_size:
            observation = env.reset()
            action = env.sample_action_space()
            observation_, reward, done, info = env.step(action)
            env.remember(observation, action, reward, observation_, done)
            n_steps += 1
        env.learn()
        env.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        observation = env.reset()
        score = 0
        done_n = [False for _ in range(env.n_agents)]
        while not all(done_n):

            action = env.choose_action(observation, evaluate)
            print("Action type", action)
            observation_, reward, done_n, info = env.step(action)
            score += sum(reward)

            if not load_checkpoint:
                env.remember(observation, action, sum(reward), observation_, done_n)
                env.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])
        obs_score_history.append(info)
        avg_obs_score = np.mean(obs_score_history[-1000:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                env.save_models()
        eps_history.append(env.epsilon)
        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'avg obs score %.1f' % avg_obs_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plotLearning(x, score_history, eps_history, figure_file)
        plotLearning(x, obs_score_history,eps_history,observation_file)
 
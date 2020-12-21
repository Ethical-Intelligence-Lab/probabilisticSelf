from pdb import set_trace
import sys
import argparse
from self_model import Self_class

import gym
import gym_gridworld
import pickle
from utils.keys import key_converter

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.callbacks import BaseCallback

parser = argparse.ArgumentParser(description='Process input args.')
parser.add_argument('-p', '--player', type=str, help='game player', metavar='')         
args = parser.parse_args()

env = gym.make('gridworld-v0')
self_class = Self_class() 
env.verbose = True
env.player = args.player
#env = gym.make('CartPole-v1')

obs = env.reset()
steps = []
step_counter = 0
n_timesteps = 25000
for i in range(n_timesteps): #25000
    if args.player == 'dqn_training':
        model = DQN(MlpPolicy, env, verbose=1, learning_rate=0.00025)
        model.learn(total_timesteps=n_timesteps)
        model.save("models/deepq_gridworld")
    elif args.player == 'dqn_trained':
        model = DQN.load("models/deepq_gridworld", env, verbose=1)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env._render()
    elif args.player == 'random':
        obs, reward, done, info = env.step(env.action_space.sample()) 
        if done:
            env.reset()
    elif args.player == 'human':
        while True:
            prelim_action = input('Enter next action: ')
            if prelim_action in ['w', 'a', 's', 'd']: 
                action = key_converter(prelim_action)
                break
            else:
                print("Please enter a valid key (w, a, s, or d).")
                continue
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()
    elif args.player == 'self_class':
        action = self_class.predict(env)
        obs, reward, done, info = env.step(action) 
        if done:
            env.reset()
    step_counter += 1


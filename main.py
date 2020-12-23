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

from utils.neptune_creds import *
#from params.default_params import PARAMS
#import neptune

# Get arguments
parser = argparse.ArgumentParser(description='Process input args.')
parser.add_argument('-p', '--player', type=str, help='game player', metavar='', required=True)        
parser.add_argument('-exp', '--experiment', type=str, help='experiment', metavar='', required=True)

args = parser.parse_args()

# Create experiment
# neptune.init(project_qualified_name='juliandefreitas/proba-self123',
#              api_token=NEPTUNE_API_TOKEN,
#              )

# exp_neptune = neptune.create_experiment(name=args.experiment,
#                                 upload_source_files=['main.py','gridworld_env.py','default_params.py'],
#                                 params=PARAMS)

# Initilize environment and self class
env = gym.make('gridworld-v0') #'CartPole-v1'
env.player = args.player
env.verbose = True
env.exp_name = args.experiment
#env.exp_neptune = exp_neptune
self_class = Self_class() 

# Model params
obs = env.reset()
steps = []
step_counter = 0
n_timesteps = 25000
for i in range(n_timesteps): #25000
    if args.player == 'dqn_training':
        #model = DQN(MlpPolicy, env, **PARAMS)
        model = DQN(MlpPolicy, env, verbose=1, learning_rate=0.00025, gamma=0.999)
        model.learn(total_timesteps=n_timesteps)
        model.save("models/deepq_gridworld")
    elif args.player == 'dqn_trained':
        model = DQN.load("models/deepq_gridworld", env, verbose=1)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env._render()
    elif args.player == 'random':
        obs, reward, done, info = env.step(env.action_space.sample()) 
        env._render()
        if done:
            env.reset()
    elif args.player == 'human':
        while True:
            prelim_action = input('Enter next action (w=up, s=down, a=left, d=right): ')
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


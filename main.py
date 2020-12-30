from pdb import set_trace
import sys
import argparse
from self_model import Self_class

import gym_l.gym as gym
import gym_gridworld
import pickle
from utils.keys import key_converter

from baselines_l.stable_baselines.common.vec_env import DummyVecEnv
from baselines_l.stable_baselines.deepq.policies import MlpPolicy
from baselines_l.stable_baselines import DQN
from baselines_l.stable_baselines.common.callbacks import BaseCallback

from utils.neptune_creds import *
from params.default_params import default_params, update_params, get_cmd_line_args
import neptune

if __name__ == '__main__':

    # Get cmd line arguments, and integrate with default params
    cmd_line_args = get_cmd_line_args(default_params) #process any inputted arguments
    P = update_params(default_params, cmd_line_args) #update default params with (optional) input arguments

    # Initilize env and Self Class
    env_id = 'gridworld-v0'
    env = gym.make('gridworld-v0') #'CartPole-v1' #'FetchSlide-v1'
    env.make_game(difficulty=P['difficulty'], player=P['player'], exp_name=P['exp_name'], singleAgent=P['singleAgent'], verbose = P['verbose'])
    self_class = Self_class() #adapt to take different games as inputs

    # Data logging to neptune AI
    if P['log_neptune']:
        neptune.init(project_qualified_name='juliandefreitas/proba-self123',
                    api_token=NEPTUNE_API_TOKEN,
                    )
        exp_neptune = neptune.create_experiment(name=args.experiment,
                                                upload_source_files=['main.py','gridworld_env.py','default_params.py'],
                                                params=P)
        env.log_neptune = True
        env.exp_neptune = exp_neptune
    
    # Main loop
    obs = env.reset()
    steps = []
    step_counter = 0
    n_timesteps = 25000
    for i in range(n_timesteps): 
        if P['player'] == 'dqn_training':
            model = DQN("MlpPolicy", env, verbose=P['verbose'], learning_rate=P['learning_rate'], gamma=P['gamma'], target_network_update_freq = P['target_network_update_freq'])
            model.learn(total_timesteps=n_timesteps)
            model.save("models/deepq_gridworld")
        elif P['player'] == 'dqn_trained':
            model = DQN.load("models/deepq_gridworld", env, verbose=P['verbose'])
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env._render()
        elif P['player'] == 'random':
            obs, reward, done, info = env.step(env.action_space.sample()) 
            env._render()
            if done:
                env.reset()
        elif P['player'] == 'human':
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
        elif P['player'] == 'self_class':
            action = self_class.predict(env)
            obs, reward, done, info = env.step(action) 
            if done:
                env.reset()
        step_counter += 1


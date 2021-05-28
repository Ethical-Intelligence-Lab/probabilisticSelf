import os
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
from baselines_l.stable_baselines import DQN, PPO2, TRPO, GAIL, HER, SAC, TD3, ACKTR, A2C
from baselines_l.stable_baselines.common.callbacks import BaseCallback

from utils.neptune_creds import *
from params.default_params import default_params, update_params, get_cmd_line_args
import neptune

if __name__ == '__main__':

    # Get cmd line arguments, and integrate with default paramss
    cmd_line_args = get_cmd_line_args(default_params) #process any inputted arguments
    P = update_params(default_params, cmd_line_args) #update default params with (optional) input arguments

    # Initilize env and Self Class
    env_id = P['env_id']
    env = gym.make(P['env_id']) #'CartPole-v1' #'FetchSlide-v1'
    env.make_game(P)

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
    loaded = False
    step_counter = 0
    n_timesteps = 1000000 #25000
    for i in range(n_timesteps): 
        if P['player'] == 'dqn_training':
            model = DQN("MlpPolicy", env, verbose=P['verbose'], learning_rate=P['learning_rate'], gamma=P['gamma'],
                        target_network_update_freq = P['target_network_update_freq'], seed=env._seed )

            path = "saved_models" + P['data_save_dir'].split("dqn_training", 1)[1]

            if not os.path.exists(path):
                os.makedirs(path)

            model.learn(total_timesteps=n_timesteps)
            model.save(path)
        elif P['player'] == 'ppo2_training':

            model = PPO2("MlpPolicy", env, verbose=P['verbose'], learning_rate=P['learning_rate'], gamma=P['gamma'], seed=env._seed)
            path = "saved_models" + P['data_save_dir'].split("ppo2_training", 1)[1]

            if not os.path.exists(path):
                os.makedirs(path)

            model.learn(total_timesteps=n_timesteps)
            model.save(path)

        elif P['player'] == 'trpo_training':

            model = TRPO("MlpPolicy", env, verbose=P['verbose'], gamma=P['gamma'], seed=env._seed)
            path = "saved_models" + P['data_save_dir'].split("trpo_training", 1)[1]

            if not os.path.exists(path):
                os.makedirs(path)

            model.learn(total_timesteps=n_timesteps)
            model.save(path)
        elif P['player'] == 'gail_training':

            model = GAIL("MlpPolicy", env)
            path = "saved_models" + P['data_save_dir'].split("gail_training", 1)[1]

            if not os.path.exists(path):
                os.makedirs(path)

            model.learn(total_timesteps=n_timesteps)
            model.save(path)
        elif P['player'] == 'her_training':

            model = HER("MlpPolicy", env, DQN)
            path = "saved_models" + P['data_save_dir'].split("her_training", 1)[1]

            if not os.path.exists(path):
                os.makedirs(path)

            model.learn(total_timesteps=n_timesteps)
            model.save(path)
        elif P['player'] == 'sac_training':

            model = SAC("MlpPolicy", env, learning_rate=P['learning_rate'], verbose=P['verbose'], gamma=P['gamma'], seed=env._seed)
            path = "saved_models" + P['data_save_dir'].split("sac_training", 1)[1]

            if not os.path.exists(path):
                os.makedirs(path)

            model.learn(total_timesteps=n_timesteps)
            model.save(path)
        elif P['player'] == 'td3_training':

            model = TD3("MlpPolicy", env, learning_rate=P['learning_rate'], verbose=P['verbose'], gamma=P['gamma'], seed=env._seed)
            path = "saved_models" + P['data_save_dir'].split("td3_training", 1)[1]

            if not os.path.exists(path):
                os.makedirs(path)

            model.learn(total_timesteps=n_timesteps)
            model.save(path)
        elif P['player'] == 'acktr_training':

            model = ACKTR("MlpPolicy", env, learning_rate=P['learning_rate'], verbose=P['verbose'], gamma=P['gamma'], seed=env._seed)
            path = "saved_models" + P['data_save_dir'].split("acktr_training", 1)[1]

            if not os.path.exists(path):
                os.makedirs(path)

            model.learn(total_timesteps=n_timesteps)
            model.save(path)
        elif P['player'] == 'a2c_training':

            model = A2C("MlpPolicy", env, learning_rate=P['learning_rate'], verbose=P['verbose'], gamma=P['gamma'], seed=env._seed)
            path = "saved_models" + P['data_save_dir'].split("a2c_training", 1)[1]

            if not os.path.exists(path):
                os.makedirs(path)

            model.learn(total_timesteps=n_timesteps)
            model.save(path)
        elif P['player'] == 'dqn_trained':
            path = "saved_models" + P['data_save_dir'].split("dqn_trained", 1)[1] + ".zip"
            if not loaded:
                model = DQN.load(path, env, verbose=P['verbose'])
                loaded = True

            model.learn(total_timesteps=n_timesteps)
            model.save(path)
        elif P['player'] == 'random':
            obs, reward, done, info = env.step(env.action_space.sample()) 
            env._render()
            if done:
                print('done')
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


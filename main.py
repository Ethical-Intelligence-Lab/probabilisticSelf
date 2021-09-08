import os
import sys
import importlib
from self_model import Self_class
import gym_l.gym as gym
import gym_gridworld
from utils.keys import key_converter
from stable_baselines import DQN, PPO2, TRPO, GAIL, HER, ACKTR, A2C, ACER
from stable_baselines.common.cmd_util import make_vec_env
from params.default_params import DefaultParams, get_cmd_line_args
import neptune.new as neptune
from dotenv import load_dotenv

import custom_callback

if __name__ == '__main__':
    load_dotenv()  # take environment variables from .env.

    arg_string = "python main.py"
    for i, arg in enumerate(sys.argv):
        if i != 0:
            arg_string = arg_string + " " + arg

    print(arg_string)

    # Get cmd line arguments, and integrate with default params
    args = get_cmd_line_args()
    player = args['player'].split('_')[0].upper() if args['player'] not in ['human', 'self_class', 'random'] else args[
        'player']
    def_params = DefaultParams(player)
    def_params.update_params(args)
    P = def_params.params

    # Initilize env and Self Class
    if P['agent_location_random'] is False and P['game_type'] != "contingency":
        print("Not implemented yet")
        exit(0)
    env_id = P['env_id']
    env = gym.make(P['env_id'])
    env.make_game(P)

    self_class = Self_class()

    run = None

    # Data logging to neptune AI
    if P['log_neptune']:
        run = neptune.init(project='akaanug/Probabilistic-Self',
                           api_token=os.environ['NEPTUNE_API_TOKEN'])  # your credentials
        run["model/parameters"] = P
        run["model/param_string"] = arg_string

    # Main loop
    obs = env.reset()
    steps = []

    if P['timestamp'] == -1:
        load_path = P['save_path'] + "lastSave/weights.zip"
    else:
        load_path = P['save_path'] + str(P['timestamp']) + "k/weights.zip"

    algo_params = def_params.get_algorithm_params()
    while True:
        if P['player'] in ['human', 'self_class', 'random']:
            if P['player'] == 'random':
                while True:
                    obs, reward, done, info = env.step(env.action_space.sample())
                    env._render()
                    if done:
                        print('done')
                        env.reset()
            elif P['player'] == 'human':
                while True:
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
                while True:
                    action = self_class.predict(env)
                    obs, reward, done, info = env.step(action)
                    if done:
                        env.reset()
            exit(0)


        def class_for_name(module_name, class_name):
            # load the module, will raise ImportError if module cannot be loaded
            m = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            c = getattr(m, class_name)
            return c


        # Run RL algorithms with either SB2 or SB3
        if P['baselines_version'] == 2:  # Stable Baselines 2
            ALGO = class_for_name('stable_baselines', player)
        elif P['baselines_version'] == 3:  # Stable Baselines 3
            print("Not supported yet.")
            exit(1)
            ALGO = class_for_name('stable_baselines3', player)
        else:
            ALGO = None
            print("Incorrect \"baselines_version\" parameter: Should be either 2 or 3.")
            exit(1)

        if player == 'A2C':
            env = make_vec_env(lambda: env, n_envs=1)  # Vectorize the environment

        if not P['load']:  # Train From Zero
            model = ALGO(def_params.get_policy(), env, **algo_params)
        else:  # Continue Training on Saved Model
            model = ALGO.load(load_path, env, verbose=P['verbose'])

        env.set_model(model)
        if P['save']:
            model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P), run=run)
        else:
            model.learn(total_timesteps=P['n_timesteps'], run=run)
        print("Training Finished. Exiting...")
        if run:
            run.stop()
        sys.exit(0)

import os
import sys
import importlib
from self_model import Self_class
import gym_l.gym as gym
import gym_gridworld  # Registering our environment here
from utils.keys import key_converter
from stable_baselines import DQN, PPO2, TRPO, GAIL, HER, ACKTR, A2C, ACER
from stable_baselines3 import DQN as DQN3, A2C as A2C3, PPO as PPO3
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env as make_vec_env3
from params.default_params import DefaultParams, get_cmd_line_args
import neptune.new as neptune
from dotenv import load_dotenv
import tensorflow as tf
import custom_callback

# Suppresses warnings about future deprecation. These warnings mostly appear because we are using
# an older version of tensorflow.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    load_dotenv()  # take environment variables from .env.

    arg_string = "python main.py"
    baselines_v = 99999
    for i, arg in enumerate(sys.argv):
        if i != 0:
            arg_string = arg_string + " " + arg

        if arg in ['-baselines_version', '--baselines_version']:
            baselines_v = i + 1

        if baselines_v == i:
            if arg.isnumeric() and (arg in ['2', '3']):
                baselines_v = int(arg)
            else:
                print("Baseline version should be an integer, i.e., 2 or 3")
                exit(1)

    print(arg_string)

    # Get cmd line arguments, and integrate with default params
    args = get_cmd_line_args(baselines_v)
    player = args['player'].split('_')[0].upper() if args['player'] not in ['human', 'self_class', 'random'] else args[
        'player']
    def_params = DefaultParams(player, baselines_v)
    def_params.update_params(args)
    P = def_params.params

    # Initialize env and Self Class
    if P['agent_location_random'] is False and P['game_type'] != "contingency":
        print("Not implemented yet")
        exit(0)
    env_id = P['env_id']
    env = gym.make(P['env_id'])
    env.seed(P['seed'])
    env.make_game(P)
    tf.compat.v1.set_random_seed(P['seed'])

    self_class = Self_class(P['seed'])

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

    if algo_params.get('n_cpu_tf_sess', False) != False:
        print("N-cpu = 1")
        algo_params['n_cpu_tf_sess'] = 1
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
            ALGO = class_for_name('stable_baselines3', player)
        else:
            ALGO = None
            print("Incorrect \"baselines_version\" parameter: Should be either 2 or 3.")
            exit(1)

        original_env = None
        if player in ['A2C', 'ACER', 'PPO2']:
            original_env = env

            if baselines_v == 2:
                env = make_vec_env(lambda: env, n_envs=1, seed=P['seed'])  # Vectorize the environment
            else:
                env = make_vec_env3(env, n_envs=1, seed=P['seed'])

        if not P['load']:  # Train From Zero
            model = ALGO(def_params.get_policy(), env, **algo_params)
        else:  # Continue Training on Saved Model
            model = ALGO.load(load_path, env, verbose=P['verbose'])
            if P['save_and_load_replay_buffer']:
                model.load_replay_buffer(P['save_path'] + "replay_buffer")

        #if original_env:
        #    original_env.set_model(model)
        #env.set_model(model)
        if P['save']:
            if P['baselines_version'] == 2:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P), run=run)
            else:
                model.learn(total_timesteps=P['n_timesteps'])
        else:
            if P['baselines_version'] == 2:
                model.learn(total_timesteps=P['n_timesteps'], run=run)
            else:
                model.learn(total_timesteps=P['n_timesteps'])
        print("Training Finished. Exiting...")
        if run:
            run.stop()
        sys.exit(0)

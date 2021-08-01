from pprint import pprint

from self_model import Self_class
import gym_l.gym as gym
import gym_gridworld
from utils.keys import key_converter
from stable_baselines import DQN, PPO2, TRPO, GAIL, HER, ACKTR, A2C, ACER
from stable_baselines.common.cmd_util import make_vec_env
from utils.neptune_creds import *
from params.default_params import default_params, update_params, get_cmd_line_args
import neptune

import custom_callback

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

    game_name_pf = "_game_shuffled/" if P['shuffle_keys'] else "_game/"
    load_path = 'saved_models/' + P['load_game'] + game_name_pf + P['player'] + '/' + \
           "seed" + str(P['seed']) + "/lr" + str(P['learning_rate']) + "_gamma" + str(P['gamma']) + \
           "_ls" + str(P['learning_starts']) + '_s' + \
           str(int(P['shuffle_keys'])) + "_prio" + str(int(P['prioritized_replay'])) + "_"

    n_timesteps = 50000 if P['player'] in ["dqn_training", "acer_training"] else 1000000000
    while True:
        print("... STARTING ITERATION ...")
        if P['player'] == 'dqn_training' and not P['load']:
            print("Seed: ", P['seed'])
            model = DQN("MlpPolicy", env, verbose=P['verbose'], learning_rate=P['learning_rate'], gamma=P['gamma'], prioritized_replay=P['prioritized_replay'],
                        target_network_update_freq=P['target_network_update_freq'], seed=P['seed'], batch_size=P['batch_size'])  # tensorboard_log="./tensorboard_results/dqn_tensorboard/")  # tensorboard_log="./tensorboard_results/dqn_tensorboard/"
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'ppo2_training' and not P['load']:
            model = PPO2("MlpPolicy", env, verbose=P['verbose'], learning_rate=P['learning_rate'], gamma=P['gamma'], seed=P['seed'])  #tensorboard_log="./tensorboard_results/ppo2_tensorboard/"
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'trpo_training' and not P['load']:
            model = TRPO("MlpPolicy", env, verbose=P['verbose'], gamma=P['gamma'], seed=P['seed'], )  #tensorboard_log="./tensorboard_results/trpo_tensorboard/"
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'gail_training' and not P['load']:
            model = GAIL("MlpPolicy", env)
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'her_training' and not P['load']:  #  TODO: How to convert our environment to goal environment? (AttributeError: 'Box' object has no attribute 'spaces')
            model = HER("MlpPolicy", env, model_class=DQN)
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'acktr_training' and not P['load']:
            model = ACKTR("MlpPolicy", env, learning_rate=P['learning_rate'], verbose=P['verbose'], gamma=P['gamma'], seed=P['seed'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'a2c_training' and not P['load']:
            model = A2C("MlpPolicy", env, learning_rate=P['learning_rate'], verbose=P['verbose'], gamma=P['gamma'],
                        seed=P['seed'], n_steps=P['n_steps'])  #tensorboard_log="./tensorboard_results/a2c_tensorboard/"
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'acer_training' and not P['load']:
            model = ACER("MlpPolicy", env, learning_rate=P['learning_rate'], verbose=P['verbose'], gamma=P['gamma'],
                         seed=P['seed'], n_steps=P['n_steps'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        # LOAD
        elif P['player'] == 'dqn_training' and P['load']:  # Play with loaded DQN agent
            path = load_path + str(P['timestamp']) + "k/weights.zip"
            if P['timestamp'] == -1:
                path = load_path + "lastSave/weights.zip"

            model = DQN.load(path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'a2c_training' and P['load']:  # Play with loaded DQN agent
            path = load_path + str(P['timestamp']) + "k/weights.zip"
            if P['timestamp'] == -1:
                path = load_path + "lastSave/weights.zip"

            env = make_vec_env(lambda: env, n_envs=1)  # Vectorize the environment
            model = A2C.load(path, env, verbose=P['verbose'])
            env.set_model(model)

            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'trpo_training' and P['load']:  # Play with loaded DQN agent
            path = load_path + str(P['timestamp']) + "k/weights.zip"
            if P['timestamp'] == -1:
                path = load_path + "lastSave/weights.zip"

            model = TRPO.load(path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'ppo2_training' and P['load']:  # Play with loaded DQN agent
            path = load_path + str(P['timestamp']) + "k/weights.zip"
            if P['timestamp'] == -1:
                path = load_path + "lastSave/weights.zip"

            model = PPO2.load(path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'acktr_training' and P['load']:  # Play with loaded DQN agent
            path = load_path + str(P['timestamp']) + "k/weights.zip"
            if P['timestamp'] == -1:
                path = load_path + "lastSave/weights.zip"

            model = ACKTR.load(path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'acer_training' and P['load']:  # Play with loaded DQN agent
            path = load_path + str(P['timestamp']) + "k/weights.zip"
            if P['timestamp'] == -1:
                path = load_path + "lastSave/weights.zip"

            env = make_vec_env(lambda: env, n_envs=1)  # Vectorize the environment
            model = ACER.load(path, env, verbose=P['verbose'])
            env.set_model(model)

            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'her_training' and P['load']:  # Play with loaded DQN agent
            path = load_path + str(P['timestamp']) + "k/weights.zip"
            if P['timestamp'] == -1:
                path = load_path + "lastSave/weights.zip"

            model = HER.load(path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

        elif P['player'] == 'gail_training' and P['load']:  # Play with loaded DQN agent
            path = load_path + str(P['timestamp']) + "k/weights.zip"
            if P['timestamp'] == -1:
                path = load_path + "lastSave/weights.zip"

            model = GAIL.load(path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=n_timesteps, callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=n_timesteps)

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



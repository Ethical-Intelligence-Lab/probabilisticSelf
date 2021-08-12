from self_model import Self_class
import gym_l.gym as gym
import gym_gridworld
from utils.keys import key_converter
from stable_baselines import DQN, PPO2, TRPO, GAIL, HER, ACKTR, A2C, ACER
from stable_baselines.common.cmd_util import make_vec_env
from utils.neptune_creds import *
from params.default_params import DefaultParams, get_cmd_line_args
import neptune

import custom_callback

if __name__ == '__main__':

    # Get cmd line arguments, and integrate with default params
    args = get_cmd_line_args()
    player = args['player'].split('_')[0].upper() if args['player'] not in ['human', 'self_class', 'random'] else args['player']
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

    # Data logging to neptune AI
    #if P['log_neptune']:
    #    neptune.init(project_qualified_name='juliandefreitas/proba-self123',
    #                 api_token=NEPTUNE_API_TOKEN,
    #                 )
    #    exp_neptune = neptune.create_experiment(name=args.experiment,
    #                                            upload_source_files=['main.py', 'gridworld_env.py',
    #                                                                 'default_params.py'],
    #                                            params=P)
    #    env.log_neptune = True
    #    env.exp_neptune = exp_neptune

    # Main loop
    obs = env.reset()
    steps = []


    if P['timestamp'] == -1:
        load_path = P['save_path'] + "lastSave/weights.zip"
    else:
        load_path = P['save_path'] + str(P['timestamp']) + "k/weights.zip"

    algo_params = def_params.get_algorithm_params()
    while True:
        #print("... STARTING ITERATION ...")
        if P['player'] == 'dqn_training' and not P['load']:
            model = DQN(def_params.get_policy(), env, **algo_params)
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'ppo2_training' and not P['load']:
            model = PPO2(def_params.get_policy(), env, **algo_params)
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'trpo_training' and not P['load']:
            model = TRPO(def_params.get_policy(), env, **algo_params)
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'gail_training' and not P['load']:
            model = GAIL(def_params.get_policy(), env, **algo_params)
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'her_training' and not P[
            'load']:  # TODO: How to convert our environment to goal environment? (AttributeError: 'Box' object has no attribute 'spaces')
            model = HER(def_params.get_policy(), env, **algo_params)
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'acktr_training' and not P['load']:
            model = ACKTR(def_params.get_policy(), env, **algo_params)
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'a2c_training' and not P['load']:
            model = A2C(def_params.get_policy(), env, **algo_params)
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'acer_training' and not P['load']:
            model = ACER(def_params.get_policy(), env, **algo_params)
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        # LOAD
        elif P['player'] == 'dqn_training' and P['load']:  # Play with loaded DQN agent
            model = DQN.load(load_path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'a2c_training' and P['load']:  # Play with loaded DQN agent
            env = make_vec_env(lambda: env, n_envs=1)  # Vectorize the environment
            model = A2C.load(load_path, env, verbose=P['verbose'])
            env.set_model(model)

            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'trpo_training' and P['load']:  # Play with loaded DQN agent
            model = TRPO.load(load_path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'ppo2_training' and P['load']:  # Play with loaded DQN agent
            model = PPO2.load(load_path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'acktr_training' and P['load']:  # Play with loaded DQN agent
            model = ACKTR.load(load_path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'acer_training' and P['load']:  # Play with loaded DQN agent
            env = make_vec_env(lambda: env, n_envs=1)  # Vectorize the environment
            model = ACER.load(load_path, env, verbose=P['verbose'])
            env.set_model(model)

            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'her_training' and P['load']:  # Play with loaded DQN agent
            model = HER.load(load_path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

        elif P['player'] == 'gail_training' and P['load']:  # Play with loaded DQN agent
            model = GAIL.load(load_path, env, verbose=P['verbose'])
            env.set_model(model)
            if P['save']:
                model.learn(total_timesteps=P['n_timesteps'], callback=custom_callback.CustomCallback(P))
            else:
                model.learn(total_timesteps=P['n_timesteps'])

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

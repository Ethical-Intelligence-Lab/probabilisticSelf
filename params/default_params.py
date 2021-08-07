import random
import string
from pdb import set_trace
import argparse

# defaults are in comments
default_params = {
    # mod params
    'gamma': 0.9,  # 0.99
    'learning_rate': 0.00025,  # 0.0005
    'buffer_size': 50000,  # 50000
    'exploration_fraction': 0.1,  # 0.1
    'exploration_final_eps': 0.02,  # 0.02
    'exploration_initial_eps': 1.0,  # 1.0
    'train_freq': 1,  # 1
    'batch_size': 32,  # 32
    'double_q': True,  # True
    'learning_starts': 1000,  # 1000
    'target_network_update_freq': 500,  # 500
    'prioritized_replay': True,  # False
    'prioritized_replay_alpha': 0.6,  # 0.6
    'prioritized_replay_beta0': 0.4,  # 0.4
    'prioritized_replay_beta_iters': None,  # None
    'prioritized_replay_eps': 1e-06,  # 1e-06
    'param_noise': False,  # False
    'n_cpu_tf_sess': None,  # None
    'tensorboard_log': None,  # None
    '_init_setup_model': True,  # True
    'policy_kwargs': None,  # None
    'full_tensorboard_log': False,  # False
    'seed': 0,

    # env params
    'env_id': 'gridworld-v0',
    'singleAgent': False,
    'game_type': 'logic',  # logic or contingency or change_agent
    'player': 'random',  # random, human, dqn_training, self_class, ppo2_training
    'exp_name': 'train_',
    'verbose': False,
    'single_loc': False,
    'n_levels': 100,
    'shuffle_keys': False,

    # data params !add 'data_save_dir'
    'log_neptune': False,
    'data_save_dir': None,
    'load': False,  # Load pretrained agent
    'timestamp': 100,  # Select which weight to run. Enter -1 for the latest.
    'save': True,  # Save the weights
    'levels_count': 20,  # Stop until 100 * 'levels_count' levels
    'load_game': None,  # Which weights to load
    'n_steps': 10,
    'agent_location_random': True,  # Is agent location random or not
}


def update_params(params, arguments):
    for arg in arguments.keys():
        if arguments[arg] is not None:
            params[arg] = arguments[arg]

    if params['load_game'] is None:
        params['load_game'] = params['game_type']

    params['data_save_dir'] = 'data/' + params['game_type'] + '_game/' + params['player'] + '/' + \
                              "seed" + str(params['seed']) + "_lr" + str(params['learning_rate']) + "_gamma" + str(
        params['gamma']) + \
                              "_ls" + str(params['learning_starts']) + '_s' + \
                              str(int(params['shuffle_keys'])) + "_prio" + str(int(params['prioritized_replay'])) + \
                              "load" + str(int(params['load'])) + "_" + "n_steps" + str(params['n_steps']) + "/"

    if params['player'] == 'random' or params['player'] == 'self_class':
        params['data_save_dir'] = 'data/' + params['game_type'] + '_game/' + params['player'] + '/' + "iter" + str(
            params['seed']) + '/'

    if params['player'] == 'human':
        letters = string.digits
        rand_id = ''.join(random.choice(letters) for i in range(10))
        params['data_save_dir'] = 'data/' + params['game_type'] + '_game/' + params['player'] + '/' + "player" + str(
            rand_id) + '/'

    if params['shuffle_keys']:
        if params['player'] == 'random' or params['player'] == 'self_class':
            params['data_save_dir'] = 'data/' + params['game_type'] + '_game_shuffled/' + params[
                'player'] + '/' + "iter" + str(
                params['seed']) + '/'
        else:
            params['data_save_dir'] = 'data/' + params['game_type'] + '_game_shuffled/' + params['player'] + '/' + \
                                      "seed" + str(params['seed']) + "_lr" + str(
                params['learning_rate']) + "_gamma" + str(params['gamma']) + \
                                      "_ls" + str(params['learning_starts']) + '_s' + \
                                      str(int(params['shuffle_keys'])) + "_prio" + str(
                int(params['prioritized_replay'])) + \
                                      "load" + str(int(params['load'])) + "/"

    return params

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_cmd_line_args(params):
    parser = argparse.ArgumentParser()
    for param_name in params.keys():
        if params[param_name] == None:
            input_type = str
        else:
            if type(params[param_name]) == bool:
                input_type = str2bool
            else:
                input_type = type(params[param_name])
        parser.add_argument("-" + param_name, "--" + param_name, type=input_type, required=False)

    cmd_line_params = vars(parser.parse_args())

    return cmd_line_params

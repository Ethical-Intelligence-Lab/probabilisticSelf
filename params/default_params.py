import random
import string
import argparse
import inspect

try:
    from stable_baselines import PPO2, DQN, TRPO, GAIL, HER, ACKTR, A2C, ACER
    from stable_baselines3 import DQN as DQN3, A2C as A2C3, PPO as PPO3
    sb_available = True
except ModuleNotFoundError as err:
    print(err)
    sb_available = False

from params.param_dicts import param_abbreviations, params
import os, sys


class DefaultParams:
    def __init__(self, player, baselines_v):
        self.player = player
        self.param_abbreviations = param_abbreviations
        self.params = params
        self.params['player'] = player
        self.params['save_path'] = ''

        if baselines_v == -1:
            self.algo_only = {}
            return

        try:
            from stable_baselines import PPO2, DQN, TRPO, GAIL, HER, ACKTR, A2C, ACER
            from stable_baselines3 import DQN as DQN3, A2C as A2C3, PPO as PPO3
        except ModuleNotFoundError as err:
            print(err)

        if player not in ['human', 'self_class', 'random']:
            # Get arguments of particular algorithm
            if baselines_v == 3 and self.params['player'] == 'DQN':
                self.params['player'] = 'DQN3'
            elif baselines_v == 3 and self.params['player'] == 'A2C':
                self.params['player'] = 'A2C3'
            elif baselines_v == 3 and self.params['player'] == 'PPO':
                self.params['player'] = 'PPO3'

            param_names = inspect.getfullargspec(globals()[self.params['player']])[0][1:]
            param_defaults = inspect.getfullargspec(globals()[self.params['player']])[3]
            param_names = [name for name in param_names if name not in ['env', 'model_class', 'policy']]
        else:
            param_names = []
            param_defaults = []

        self.algo_only = {}

        for i, name in enumerate(param_names):
            self.params[name] = param_defaults[i]
            self.algo_only[name] = param_defaults[i]

    def get_algorithm_params(self):
        return self.algo_only

    def get_policy(self):
        return self.params['policy']

    def update_params(self, arguments):
        for arg in arguments.keys():
            if arguments[arg] is not None and arg in params:
                self.params[arg] = arguments[arg]

            if arguments[arg] is not None and arg in self.algo_only:
                self.algo_only[arg] = arguments[arg]

        # self.algo_only['policy'] = arguments['policy'] if arguments['policy'] is not None else self.params['policy']
        self.algo_only['seed'] = arguments['seed'] if arguments['seed'] is not None else self.params['seed']

        if self.params['load_game'] is None:
            self.params['load_game'] = self.params['game_type']

        root_path = ''
        if self.params['use_scratch_space']:
            root_path = '/export/scratch/auguralp/'

        game_str = "_game_shuffled_{}".format(self.params['shuffle_each']) if self.params['shuffle_keys'] else "_game"
        if self.params['different_self_color']:
            game_str = game_str + "_diff_color"

        if self.params['modify_to'] == 'switching_embodiments_extended_2':
            game_str = game_str + '_self_finding'

        game_str = game_str + "-agent_loc_constant/" if not self.params['agent_location_random'] else game_str + "/"
        if self.params['player'] not in ['human', 'random', 'self_class']:
            algo_params_str = ""
            for key, val in self.algo_only.items():
                if isinstance(val, bool):
                    val = 1 if val else 0

                if key in self.param_abbreviations.keys() and self.param_abbreviations[key] != '':
                    algo_params_str += self.param_abbreviations[key] + "=" + str(val) + "-"

            path_len = len(os.path.abspath(os.getcwd()))
            if (len(algo_params_str) + path_len) > 150:
                suitable_len = 170 - path_len
                algo_params_str = algo_params_str[:suitable_len]

            letters = string.ascii_uppercase
            r_str = ''.join(random.choice(letters) for i in range(10))
            load_str = "1" if self.params['load'] else "0"
            self.params['data_save_dir'] = root_path + 'data/' + self.params['game_type'] + game_str + self.params['player'] + '/' + \
                                           "seed" + str(self.params['seed']) + "-" + r_str + '-' \
                                                                                             "load=" + load_str + "-" + "n_ts=" + str(
                "{:.2e}".format(self.params['n_timesteps'])) + "/"

            # Set save path
            self.params['save_path'] = root_path + 'saved_models/' + self.params['game_type'] + game_str + self.params[
                'player'] + '/' + "seed" + str(self.params['seed']) + "-" + r_str + "-"

            if self.params['load_str'] != '' and self.params['load']:
                self.params['save_path'] = root_path + 'saved_models/' + self.params['load_game'] + game_str + self.params[
                    'player'] + '/' + \
                                           "seed" + str(self.params['seed']) + "-" + self.params['load_str'] + "-"


        elif self.params['player'] != 'human':  # Random or self class
            self.params['data_save_dir'] = root_path + 'data/' + self.params['game_type'] + game_str + \
                                           ("keep_close" if self.params['keep_all_close'] else self.params['player']) + '/' + "iter" + str(self.params['seed']) + '/'
        else:  # Human
            letters = string.digits
            rand_id = ''.join(random.choice(letters) for i in range(10))
            self.params['data_save_dir'] = root_path + 'data/' + self.params['game_type'] + game_str + self.params[
                'player'] + '/' + "player" + str(
                rand_id) + '/'

        # Create save and load folders, if DNE

        return self.params


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_algo_cmd_line_args(baselines_v):
    parser = argparse.ArgumentParser()

    dqn_version = 'DQN3' if baselines_v == 3 else 'DQN'
    a2c_version = 'A2C3' if baselines_v == 3 else 'A2C'
    ppo_version = 'PPO3' if baselines_v == 3 else 'PPO2'

    all_possible_params = {}
    if sb_available:
        all_possible_params = {"seed": 0,
                            **(dict(zip([v for v in inspect.getfullargspec(globals()['{}'.format(dqn_version)])[0][1:] if
                                            v not in ['env', 'policy']],
                                        inspect.getfullargspec(globals()['{}'.format(dqn_version)])[3]))),
                            **(dict(zip([v for v in inspect.getfullargspec(globals()['{}'.format(ppo_version)])[0][1:] if
                                            v not in ['env', 'policy']],
                                        inspect.getfullargspec(globals()['{}'.format(ppo_version)])[3]))),
                            **(dict(zip([v for v in inspect.getfullargspec(globals()['TRPO'])[0][1:] if
                                            v not in ['env', 'policy']],
                                        inspect.getfullargspec(globals()['TRPO'])[3]))),
                            **(dict(zip([v for v in inspect.getfullargspec(globals()['GAIL'])[0][1:] if
                                            v not in ['env', 'policy']],
                                        inspect.getfullargspec(globals()['GAIL'])[3]))),
                            **(dict(zip([v for v in inspect.getfullargspec(globals()['HER'])[0][1:] if
                                            v not in ['env', 'policy']],
                                        inspect.getfullargspec(globals()['HER'])[3]))),
                            **(dict(zip([v for v in inspect.getfullargspec(globals()['ACKTR'])[0][1:] if
                                            v not in ['env', 'policy']],
                                        inspect.getfullargspec(globals()['ACKTR'])[3]))),
                            **(dict(zip([v for v in inspect.getfullargspec(globals()['{}'.format(a2c_version)])[0][1:] if
                                            v not in ['env', 'policy']],
                                        inspect.getfullargspec(globals()['{}'.format(a2c_version)])[3]))),
                            **(dict(zip([v for v in inspect.getfullargspec(globals()['ACER'])[0][1:] if
                                            v not in ['env', 'policy']],
                                        inspect.getfullargspec(globals()['ACER'])[3]))),
                            **params}
    else:
        all_possible_params = params

    all_possible_params['seed'] = 0

    for param_name in all_possible_params:
        if all_possible_params[param_name] == None:
            input_type = str
        else:
            if type(all_possible_params[param_name]) == bool:
                input_type = str2bool
            else:
                input_type = type(all_possible_params[param_name])
        parser.add_argument("-" + param_name, "--" + param_name, type=input_type, required=False)

    cmd_line_params = vars(parser.parse_args())

    return cmd_line_params

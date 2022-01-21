import random
import string
import argparse
import inspect
from stable_baselines import DQN, PPO2, TRPO, GAIL, HER, ACKTR, A2C, ACER
from params.param_dicts import param_abbreviations, params
import os


class DefaultParams:
    def __init__(self, player):
        self.player = player
        self.param_abbreviations = param_abbreviations
        self.params = params
        self.params['player'] = player
        self.params['save_path'] = ''

        if player not in ['human', 'self_class', 'random']:
            # Get arguments of particular algorithm
            param_names = inspect.getfullargspec(globals()[player])[0][1:]
            param_defaults = inspect.getfullargspec(globals()[player])[3]
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

        game_str = "_game_shuffled_{}".format(self.params['shuffle_each']) if self.params['shuffle_keys'] else "_game_0"
        if self.params['different_self_color']:
            game_str = game_str + "_diff_color"
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
            self.params['data_save_dir'] = 'data/' + self.params['game_type'] + game_str + self.params['player'] + '/' + \
                                           "seed" + str(self.params['seed']) + "-" + r_str + '-' \
                                           "load=" + load_str + "-" + "n_ts=" + str(
                "{:.2e}".format(self.params['n_timesteps'])) + "/"

            # Set save path

            self.params['save_path'] = 'saved_models/' + self.params['game_type'] + game_str + self.params[
                'player'] + '/' + \
                                       "seed" + str(self.params['seed']) + "-" + r_str + "-"

            if self.params['load_str'] != '' and self.params['load']:
                self.params['save_path'] = 'saved_models/' + self.params['load_game'] + game_str + self.params[
                    'player'] + '/' + \
                                           "seed" + str(self.params['seed']) + "-" + self.params['load_str'] + "-"


        elif self.params['player'] != 'human':  # Random or self class
            self.params['data_save_dir'] = 'data/' + self.params['game_type'] + game_str + \
                                           self.params['player'] + '/' + "iter" + str(self.params['seed']) + '/'
        else:  # Human
            letters = string.digits
            rand_id = ''.join(random.choice(letters) for i in range(10))
            self.params['data_save_dir'] = 'data/' + self.params['game_type'] + game_str + self.params[
                'player'] + '/' + "player" + str(
                rand_id) + '/'

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


def get_cmd_line_args():
    parser = argparse.ArgumentParser()

    all_possible_params = {"seed": 0,
                           **(dict(zip([v for v in inspect.getfullargspec(globals()['DQN'])[0][1:] if
                                        v not in ['env', 'policy']],
                                       inspect.getfullargspec(globals()['DQN'])[3]))),
                           **(dict(zip([v for v in inspect.getfullargspec(globals()['PPO2'])[0][1:] if
                                        v not in ['env', 'policy']],
                                       inspect.getfullargspec(globals()['PPO2'])[3]))),
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
                           **(dict(zip([v for v in inspect.getfullargspec(globals()['A2C'])[0][1:] if
                                        v not in ['env', 'policy']],
                                       inspect.getfullargspec(globals()['A2C'])[3]))),
                           **(dict(zip([v for v in inspect.getfullargspec(globals()['ACER'])[0][1:] if
                                        v not in ['env', 'policy']],
                                       inspect.getfullargspec(globals()['ACER'])[3]))),
                           **params}

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

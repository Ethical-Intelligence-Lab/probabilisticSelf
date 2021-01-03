from pdb import set_trace
import argparse

#defaults are in comments
default_params = { 
          # mod params
          'gamma': 0.999, #0.99
          'learning_rate': 0.00025, #0.0005
          'buffer_size': 50000, #50000
          'exploration_fraction': 0.1, #0.1
          'exploration_final_eps': 0.02, #0.02
          'exploration_initial_eps': 1.0, #1.0
          'train_freq': 1, #1
          'batch_size': 32, #32
          'double_q': True, #True
          'learning_starts': 1000, #1000
          'target_network_update_freq': 500, #500
          'prioritized_replay': False, #False
          'prioritized_replay_alpha': 0.6, #0.6
          'prioritized_replay_beta0': 0.4, #0.4
          'prioritized_replay_beta_iters': None, #None
          'prioritized_replay_eps': 1e-06, #1e-06
          'param_noise': False, #False
          'n_cpu_tf_sess': None, #None
          'verbose': 0, #1
          'tensorboard_log': None, #None
          '_init_setup_model': True, #True
          'policy_kwargs': None, #None
          'full_tensorboard_log': False, #False
          'seed': None, #None
          
          # env params
          'env_id': 'gridworld-v0',
          'singleAgent': False,
          'game_type': 'logic', #logic or contingency
          'player': 'random', #random, human, dqn_training, self_class
          'exp_name': 'test', 
          'verbose': False,
          'single_loc': False,
          'n_levels': 100,
          'shuffle_keys': True,

          # data params !add 'data_save_dir'
          'log_neptune': False
        }      

def update_params(params, arguments):
    for arg in arguments.keys():
      if arguments[arg] != None:
        params[arg] = arguments[arg]

    params['data_save_dir'] = 'data/' + params['game_type'] + '_game/' + params['player'] + '/'

    return params

def get_cmd_line_args(params):
    parser = argparse.ArgumentParser()
    for param_name in params.keys():
        if params[param_name] == None:
            input_type = str
        else:
            input_type = type(params[param_name])
        parser.add_argument("-" + param_name, "--" + param_name, type=input_type, required=False)

    cmd_line_params = vars(parser.parse_args())

    return cmd_line_params
                                                     
    
        


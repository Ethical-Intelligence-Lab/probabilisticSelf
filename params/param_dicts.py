param_abbreviations = {
    'seed': '', 'verbose': '',
    # DQN
    'gamma': 'g', 'learning_rate': 'lr', 'buffer_size': 's_buf', 'learning_starts': 'lrst',
    'exploration_fraction': 'exf', 'exploration_final_eps': 'exp_f_eps',
    'exploration_initial_eps': 'ieps', 'train_freq': 'trfrq', 'batch_size': 'b_s',
    'double_q': 'dq', 'prioritized_replay_alpha': 'pra', 'prioritized_replay_beta0': 'prb',
    'n_timesteps': 'n_ts', 'prioritized_replay_beta_iters': 'prbi',
    'prioritized_replay_eps': 'preps', 'prioritized_replay': 'pr',
    'param_noise': 'prn', 'policy': 'plc', 'tensorboard_log': '', '_init_setup_model': '', 'policy_kwargs': '',
    'full_tensorboard_log': '', 'n_cpu_tf_sess': '', 'kwargs': '', 'target_network_update_freq': 'netuf',
    # PPO2
    'n_steps': 'n_s', 'ent_coef': 'ent', 'vf_coef': 'vf', 'max_grad_norm': 'mxgn',
    'lam': 'lam', 'nminibatches': 'nmb', 'noptepochs': 'nepoc', 'cliprange': 'cr', 'cliprange_vf': 'cr_vf',
    # TRPO
    'timesteps_per_batch': 'tspb', 'max_kl': 'maxkl', 'cg_iters': 'cgite',
    'entcoeff': 'ent', 'cg_damping': 'cgd', 'vf_stepsize': 'vfs', 'vf_iters': 'vfite',
    # GAIL
    'expert_dataset': '', 'hidden_size_adversary': 'hsa', 'adversary_entcoeff': 'aec',
    'g_step': 'g_step', 'd_step': 'd_step', 'd_stepsize': 'd_ss',
    # HER
    'model_class': 'model_class', 'n_sampled_goal': 'nsg',
    'goal_selection_strategy': 'gss', 'args': '',
    # ACKTR
    'nprocs': 'nprocs', 'vf_fisher_coef': 'vffc', 'kfac_clip': 'kfacc', 'lr_schedule': 'lrsc',
    'async_eigen_decomp': 'eigen_dec', 'kfac_update': 'kfac_upd', 'gae_lambda': 'gae_lmd',
    # A2C
    'alpha': 'alpha', 'momentum': 'mmt', 'epsilon': 'epsilon',
    # ACER
    'q_coef': 'q', 'rprop_alpha': 'rprop_a', 'rprop_epsilon': 'rprop_e', 'replay_ratio': 'rep_ratio',
    'replay_start': 'rep_start', 'correction_term': 'corr_term', 'trust_region': 'treg',
    'delta': 'delta'
}

params = {
    # env params
    'policy': 'MlpPolicy',
    'seed': 0,
    'env_id': 'gridworld-v0',
    'singleAgent': False,
    'game_type': 'logic',  # logic or contingency or change_agent
    'player': 'random',  # random, human, dqn_training, self_class, ppo2_training
    'exp_name': 'train_',
    'verbose': False,
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
    'agent_location_random': True,  # Is agent location random or not
    'n_timesteps': 1000000000,
    'single_loc': False,

    'baselines_version': 2
}
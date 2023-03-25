import numpy as np
import argparse
import torch
from copy import deepcopy
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from option_critic.option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic.option_critic import critic_loss as critic_loss_fn
from option_critic.option_critic import actor_loss as actor_loss_fn

from option_critic.experience_replay import ReplayBuffer
from option_critic.utils import make_env, to_tensor
from params.default_params import DefaultParams
from params.param_dicts import params
from gym_gridworld.envs.gridworld_env import GridworldEnv

from dotenv import load_dotenv
import neptune
import neptune.integrations.optuna as npt_utils
import sys
import os

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='gridworld-v0', help='ROM to run')
parser.add_argument('--optimal_eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame_skip', default=4, type=int, help='Every how many frames to process')
parser.add_argument('--learning_rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon_start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon_min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon_decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max_history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze_interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update_frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination_reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy_reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num_options', type=int, default=2, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(4e6), help='number of maximum steps to take.') # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')
parser.add_argument('--optimize', type=bool, default=False, help='Optimize hyperparameters')

# Parameters for gridworld:
for k, v in params.items():
    parser.add_argument('--{}'.format(str(k)), type=type(k), default=v)


def objective(trial):
    gamma = trial.suggest_float("gamma", 0.1, 0.999, log=True)
    learning_rate = trial.suggest_float("learning_rate", 0.000001, 0.1, log=True)
    n_options = trial.suggest_int("n_options", 2, 16)
    epsilon_decay = trial.suggest_int("epsilon_decay", 1000, 50000)
    epsilon_start = trial.suggest_float("epsilon_start", 0.001, 1.0, log=True)
    epsilon_min = trial.suggest_float("epsilon_min", 0.000001, 1.0)
    optimal_eps = trial.suggest_float("optimal_eps", 0.000001, 1.0, log=True)
    max_history = trial.suggest_int("max_history", 500, 30000)
    batch_size = trial.suggest_int("batch_size", 32, 256)
    freeze_interval = trial.suggest_int("freeze_interval", 1, 800)
    update_frequency = trial.suggest_int("update_frequency", 1, 20)
    termination_reg = trial.suggest_float("termination_reg", 0, 1.0)
    entropy_reg = trial.suggest_float("entropy_reg", 0, 1.0)
    temp = trial.suggest_float("temp", 0, 1)

    # Modify args
    args.epsilon_decay = epsilon_decay
    args.epsilon_start = epsilon_start
    args.epsilon_min = epsilon_min

    args.gamma = gamma
    args.learning_rate = learning_rate
    args.num_options = n_options
    args.optimal_eps = optimal_eps
    args.max_history = max_history
    args.batch_size = batch_size
    args.freeze_interval = freeze_interval
    args.update_frequency = update_frequency
    args.termination_reg = termination_reg
    args.entropy_reg = entropy_reg
    args.temp = temp

    print(args)

    return run(args, P)


def run(args, P):
    env, is_atari = make_env(args.env, P)
    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    option_critic = option_critic(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    env.seed(int(args.seed))

    buffer = ReplayBuffer(capacity=args.max_history, seed=int(args.seed))

    steps = 0 ;
    if args.switch_goal: print(f"Current goal {env.goal}")
    while steps < args.max_steps_total:

        rewards = 0 ; option_lengths = {opt:[] for opt in range(args.num_options)}

        obs   = env.reset()
        state = option_critic.get_state(to_tensor(obs))
        greedy_option  = option_critic.greedy_option(state)
        current_option = 0

        # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
        # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
        # should be finedtuned (this is what we would hope).
        if args.switch_goal:
            torch.save({'model_params': option_critic.state_dict(),
                        'goal_state': env.goal},
                        f'models/option_critic_seed={args.seed}_1k')
            env.switch_goal()
            print(f"New goal {env.goal}")

        if args.switch_goal:
            torch.save({'model_params': option_critic.state_dict(),
                        'goal_state': env.goal},
                        f'models/option_critic_seed={args.seed}_2k')
            break

        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        while not done and ep_steps < args.max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0
    
            action, logp, entropy = option_critic.get_action(state, current_option)

            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                    reward, done, next_obs, option_critic, option_critic_prime, args)
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

    return env.get_total_level_count() # Return number of levels finished

if __name__=="__main__":
    load_dotenv()  # take environment variables from .env.

    arg_string = "python main_option_critic.py"
    for i, arg in enumerate(sys.argv):
        if i != 0:
            arg_string = arg_string + " " + arg

    args = parser.parse_args()

    def_params = DefaultParams(player="feudal", baselines_v=-1)
    def_params.update_params(vars(args))
    P = def_params.params
    P['args'] = vars(args)

    if P['log_neptune']:
        run_n = neptune.init_run(project='akaanug/Probabilistic-Self',
                           api_token=os.environ['NEPTUNE_API_TOKEN'])  # your credentials
        run_n["model/parameters"] = P
        run_n["model/param_string"] = arg_string
        neptune_callback = npt_utils.NeptuneCallback(run_n)
        P['run'] = run_n

    torch.manual_seed(int(P['seed']))

    if args.optimize:
        torch.set_num_threads(1)
        N_STARTUP_TRIALS = 5
        N_EVALUATIONS = 2
        N_TRIALS = 200

        sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
        study = optuna.create_study(sampler=sampler, direction="maximize") #pruner=pruner
        try:
            study.optimize(objective, n_trials=N_TRIALS, timeout=36000, callbacks=[neptune_callback])
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print("    {}: {}".format(key, value))

        run_n.stop()
    else:
        run(args, P)

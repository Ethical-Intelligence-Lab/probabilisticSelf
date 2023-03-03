from collections import namedtuple
import os
import argparse

import torch
import torch.multiprocessing as mp

from feudal_code import my_optim
from feudal_code.fun import FeudalNet
from feudal_code.envs import create_atari_env
from feudal_code.train import train
from feudal_code.test import test
from gym_gridworld.envs.gridworld_env import GridworldEnv
from params.default_params import DefaultParams
from params.param_dicts import params
from dotenv import load_dotenv
import neptune.new as neptune
import sys

parser = argparse.ArgumentParser(description='Feudal Net with A3C setup')
parser.add_argument('--lr', type=float, default=0.00001,  # try LogUniform(1e-4.5, 1e-3.5)
                    help='learning rate')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='intrinsic reward multiplier')
parser.add_argument('--gamma-worker', type=float, default=0.95,
                    help='worker discount factor for rewards')
parser.add_argument('--gamma-manager', type=float, default=0.99,
                    help='manager discount factor for rewards')
parser.add_argument('--tau-worker', type=float, default=0.9,
                    help='parameter for GAE (worker only)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (also called beta)')
parser.add_argument('--value-worker-loss-coef', type=float, default=1,
                    help='worker value loss coefficient')
parser.add_argument('--value-manager-loss-coef', type=float, default=1,
                    help='manager value loss coefficient')
parser.add_argument('--max-grad-norm', type=float, default=0.5,
                    help='value loss coefficient')
parser.add_argument('--num-processes', type=int, default=1,
                    help='how many training processes to use')
parser.add_argument('--num-steps', type=int, default=10,
                    help='number of forward steps in A3C')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode')
parser.add_argument('--env-name', default='gridworld-v0',
                    help='environment to train on')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')

# Parameters for gridworld:
for k, v in params.items():
    parser.add_argument('--{}'.format(str(k)), type=type(k), default=v)

if __name__ == "__main__":
    load_dotenv()  # take environment variables from .env.

    arg_string = "python feudal.py"
    for i, arg in enumerate(sys.argv):
        if i != 0:
            arg_string = arg_string + " " + arg

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    mp.set_start_method('spawn')

    args = parser.parse_args()

    def_params = DefaultParams(player="feudal", baselines_v=-1)
    def_params.update_params(vars(args))
    P = def_params.params

    print(args)
    P['args'] = args

    # Data logging to neptune AI
    if P['log_neptune']:
        run = neptune.init(project='akaanug/Probabilistic-Self',
                           api_token=os.environ['NEPTUNE_API_TOKEN'])  # your credentials
        run["model/parameters"] = P
        run["model/param_string"] = arg_string
        P['run'] = run

    torch.manual_seed(P['seed'])
    env = create_atari_env(args.env_name, P=P)
    shared_model = FeudalNet(env.observation_space, env.action_space, channel_first=True)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    import socket
    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname())

    train(0, shared_model, counter, log_dir, lock, optimizer, args, P)

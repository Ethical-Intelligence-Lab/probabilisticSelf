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

parser = argparse.ArgumentParser(description='Feudal Net with A3C setup')
parser.add_argument('--lr', type=float, default=0.00001,  # try LogUniform(1e-4.5, 1e-3.5)
                    help='learning rate')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='intrinsic reward multiplier')
parser.add_argument('--gamma-worker', type=float, default=0.95,
                    help='worker discount factor for rewards')
parser.add_argument('--gamma-manager', type=float, default=0.99,
                    help='manager discount factor for rewards')
parser.add_argument('--tau-worker', type=float, default=1.00, # 0.9
                    help='parameter for GAE (worker only)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (also called beta)')
parser.add_argument('--value-worker-loss-coef', type=float, default=1,
                    help='worker value loss coefficient')
parser.add_argument('--value-manager-loss-coef', type=float, default=1,
                    help='manager value loss coefficient')
parser.add_argument('--max-grad-norm', type=float, default=40,
                    help='value loss coefficient')
parser.add_argument('--num-processes', type=int, default=1,
                    help='how many training processes to use')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode')
parser.add_argument('--env-name', default='gridworld-v0',
                    help='environment to train on')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed')

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    mp.set_start_method('spawn')

    args = parser.parse_args()

    def_params = DefaultParams(player="feudal", baselines_v=-1)
    def_params.update_params(vars(args))
    P = def_params.params

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

    p = mp.Process(target=test, args=(args.num_processes, shared_model, counter, log_dir, lock, args, P))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, shared_model, counter, log_dir, lock, optimizer, args, P))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

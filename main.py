from pdb import set_trace
import sys
import argparse

import gym
import gym_gridworld

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

parser = argparse.ArgumentParser(description='Process input args.')
parser.add_argument('-p', '--player', type=str, help='game player', metavar='')         
args = parser.parse_args()

env = gym.make('gridworld-v0')
env.verbose = True
#env = gym.make('CartPole-v1')

def key_converter(key_pressed):
    if key_pressed == 'w':
        print('UP!')
        return 1
    elif key_pressed == 's':
        print('DOWN!')
        return 2
    elif key_pressed == 'a':
        print('LEFT!')
        return 3
    elif key_pressed == 'd':
        print('RIGHT!')
        return 4

obs = env.reset()
steps = []
step_counter = 0
for i in range(25000):
    if args.player == 'dqn_untrained':
        model = DQN(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=25000)
        model.save("deepq_gridworld")
        del model 
    if args.player == 'dqn_trained':
        model = DQN.load("deepq_cartpole")
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
    elif args.player == 'random':
        obs, reward, done, info = env.step(env.action_space.sample()) 
    elif args.player == 'human':
        action = key_converter(input('Enter next action: '))
        obs, reward, done, info = env.step(action)
    step_counter += 1
    if done:
        steps.append(step_counter)
        step_counter = 0
        obs = env.reset()




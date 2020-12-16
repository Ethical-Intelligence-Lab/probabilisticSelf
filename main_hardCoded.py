import gym
import gym_gridworld
from pdb import set_trace

env = gym.make('gridworld-v0')
env.verbose = True
_ = env.reset()
while True:
    obs, rewards, dones, info = env.step(env.action_space.sample())

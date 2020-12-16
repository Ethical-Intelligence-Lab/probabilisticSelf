# import gym
# import gym_gridworld

# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN

# env = gym.make('CartPole-v0')
# env.verbose = True
# #env = gym.make('gridworld-v0')

# model = DQN(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("deepq_gridworld")

# obs = env.reset()
# env.render()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     #env.render()

import gym
import gym_gridworld

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('gridworld-v0')
env.verbose = True
#env = gym.make('CartPole-v1')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("deepq_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("deepq_cartpole")

obs = env.reset()
for i in range(25000):
    if rewards > 0:
        obs = env.reset()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

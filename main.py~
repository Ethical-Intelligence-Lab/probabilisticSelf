import gym
import gym_gridworld

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('gridworld-v0')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("deepq_cartpole")

obs = env.reset()
i = 0
while True:
    i += 1
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print('step: ', i)
    env.render()

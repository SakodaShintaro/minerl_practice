"""Test the environment with a random agent."""

import logging

import gym
import minerl  # noqa: F401

logging.basicConfig(level=logging.DEBUG)
env = gym.make("MineRLObtainDiamondShovel-v0")
obs = env.reset()
done = False

while not done:
    # Take a random action
    action = env.action_space.sample()
    action["ESC"] = 0
    obs, reward, done, _ = env.step(action)
    print(obs.keys())
    print(obs["pov"].shape, obs["pov"].dtype)
    print(action)
    env.render()
    break

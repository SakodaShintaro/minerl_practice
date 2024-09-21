"""Train a model using Stable Baselines3."""

import gym
import minerl  # noqa: F401
import numpy as np
from gym import spaces
from stable_baselines3 import A2C


class MineRLObsWrapper(gym.ObservationWrapper):
    """A wrapper for MineRL observations to extract the 'pov' key."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(360, 640, 3), dtype=np.uint8)

    def observation(self, obs: dict) -> np.ndarray:
        return np.copy(obs["pov"])


class MineRLActionWrapper(gym.ActionWrapper):
    """A wrapper to convert Dict action to MultiDiscrete for Stable-Baselines3 compatibility."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # MineRLのアクションスペースのDictをMultiDiscreteに変換
        self.action_space = spaces.MultiDiscrete(
            [
                2,  # ESC
                2,  # attack
                2,  # back
                2,  # forward
                2,  # left
                2,  # right
                2,  # jump
                2,  # sprint
                2,  # sneak
                2,  # use
                8,  # hotbar selection (9 hotbars)
                3,  # camera pitch
                3,  # camera yaw
            ],
        )

    def action(self, action: np.ndarray) -> dict:
        """Convert the MultiDiscrete action back to the Dict action format."""
        # MultiDiscrete -> Dictアクションの変換
        return {
            "ESC": action[0],
            "attack": action[1],
            "back": action[2],
            "forward": action[3],
            "left": action[4],
            "right": action[5],
            "jump": action[6],
            "sprint": action[7],
            "sneak": action[8],
            "use": action[9],
            "hotbar.1": 1 if action[10] == 0 else 0,
            "hotbar.2": 1 if action[10] == 1 else 0,
            "hotbar.3": 1 if action[10] == 2 else 0,
            "hotbar.4": 1 if action[10] == 3 else 0,
            "hotbar.5": 1 if action[10] == 4 else 0,
            "hotbar.6": 1 if action[10] == 5 else 0,
            "hotbar.7": 1 if action[10] == 6 else 0,
            "hotbar.8": 1 if action[10] == 7 else 0,
            "camera": np.array([self.decode_camera(action[11]), self.decode_camera(action[12])]),
        }

    def decode_camera(self, camera_action: int) -> float:
        """Camera action decoding (for pitch and yaw)."""
        if camera_action == 0:
            return -1.0  # Move camera down/left
        if camera_action == 1:
            return 0.0  # No camera movement
        if camera_action == 2:
            return 1.0  # Move camera up/right
        raise ValueError(f"Invalid camera action {camera_action}")


env = gym.make("MineRLObtainDiamondShovel-v0")
env = MineRLObsWrapper(env)
env = MineRLActionWrapper(env)

model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=400)

obs = env.reset()
for _ in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

"""Dummy MineRL environment for testing purposes."""

import gym
import numpy as np
from gym import spaces


class MockMineRL(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        # Action space definition matching MineRL
        self.action_space = spaces.Dict(
            {
                "ESC": spaces.Discrete(2),
                "attack": spaces.Discrete(2),
                "back": spaces.Discrete(2),
                "camera": spaces.Box(low=-180.0, high=180.0, shape=(2,)),
                "drop": spaces.Discrete(2),
                "forward": spaces.Discrete(2),
                "hotbar.1": spaces.Discrete(2),
                "hotbar.2": spaces.Discrete(2),
                "hotbar.3": spaces.Discrete(2),
                "hotbar.4": spaces.Discrete(2),
                "hotbar.5": spaces.Discrete(2),
                "hotbar.6": spaces.Discrete(2),
                "hotbar.7": spaces.Discrete(2),
                "hotbar.8": spaces.Discrete(2),
                "hotbar.9": spaces.Discrete(2),
                "inventory": spaces.Discrete(2),
                "jump": spaces.Discrete(2),
                "left": spaces.Discrete(2),
                "pickItem": spaces.Discrete(2),
                "right": spaces.Discrete(2),
                "sneak": spaces.Discrete(2),
                "sprint": spaces.Discrete(2),
                "swapHands": spaces.Discrete(2),
                "use": spaces.Discrete(2),
            },
        )

        # Observation space definition
        self.observation_space = spaces.Dict(
            {
                "pov": spaces.Box(low=0, high=255, shape=(360, 640, 3), dtype=np.uint8),
                "inventory": spaces.Dict({}),  # 簡易実装のため空のDict
            },
        )

        self.step_num = 0

    def reset(self) -> dict:
        # ランダムな画像データを生成
        self.step_num = 0
        pov = np.random.randint(0, 255, size=(360, 640, 3), dtype=np.uint8)
        return {"pov": pov, "inventory": {}}

    def step(self, _) -> tuple:  # noqa: ANN001
        # 新しい観測を生成
        obs = {
            "pov": np.random.randint(0, 255, size=(360, 640, 3), dtype=np.uint8),
            "inventory": {},
        }

        self.step_num += 1
        terminal = self.step_num >= 1000000

        return obs, 0.0, terminal, {}

    def render(self) -> None:
        pass


if __name__ == "__main__":
    env = MockMineRL()
    obs = env.reset()

    print(f"{env.action_space=}")

    action = env.action_space.sample()
    action["ESC"] = 0
    obs, reward, done, info = env.step(action)

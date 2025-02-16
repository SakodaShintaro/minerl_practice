import gym
import numpy as np
from minerl.env import _fake, _singleagent
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.mc import ALL_ITEMS

TIMEOUT = 18000
ACTION_DIM = 24


def dict_action_to_array_action(action: dict) -> np.ndarray:
    return np.array(
        [
            *action["camera"],
            action["attack"],
            action["back"],
            action["drop"],
            action["forward"],
            action["hotbar.1"],
            action["hotbar.2"],
            action["hotbar.3"],
            action["hotbar.4"],
            action["hotbar.5"],
            action["hotbar.6"],
            action["hotbar.7"],
            action["hotbar.8"],
            action["hotbar.9"],
            action["inventory"],
            action["jump"],
            action["left"],
            action["pickItem"],
            action["right"],
            action["sneak"],
            action["sprint"],
            action["swapHands"],
            action["use"],
        ],
        dtype=np.float32,
    )


def array_action_to_dict_action(action: np.ndarray) -> dict:
    return {
        "camera": action[:2],
        "attack": int(np.random.binomial(1, action[2])),
        "back": int(np.random.binomial(1, action[3])),
        "drop": int(np.random.binomial(1, action[4])),
        "forward": int(np.random.binomial(1, action[5])),
        "hotbar.1": int(np.random.binomial(1, action[6])),
        "hotbar.2": int(np.random.binomial(1, action[7])),
        "hotbar.3": int(np.random.binomial(1, action[8])),
        "hotbar.4": int(np.random.binomial(1, action[9])),
        "hotbar.5": int(np.random.binomial(1, action[10])),
        "hotbar.6": int(np.random.binomial(1, action[11])),
        "hotbar.7": int(np.random.binomial(1, action[12])),
        "hotbar.8": int(np.random.binomial(1, action[13])),
        "hotbar.9": int(np.random.binomial(1, action[14])),
        "inventory": int(np.random.binomial(1, action[15])),
        "jump": int(np.random.binomial(1, action[16])),
        "left": int(np.random.binomial(1, action[17])),
        "pickItem": int(np.random.binomial(1, action[18])),
        "right": int(np.random.binomial(1, action[19])),
        "sneak": int(np.random.binomial(1, action[20])),
        "sprint": int(np.random.binomial(1, action[21])),
        "swapHands": int(np.random.binomial(1, action[22])),
        "use": int(np.random.binomial(1, action[23])),
    }


class MySettingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0
        self.episode_over = False
        self.open_inventory = False
        self.previous_inventory_action = 0

    def step(self, action: dict) -> tuple:
        if self.episode_over:
            raise RuntimeError("Expected `reset` after episode terminated, not `step`.")
        observation, reward, done, info = super().step(action)
        if self.previous_inventory_action == 0 and action["inventory"] == 1:
            self.open_inventory = not self.open_inventory
        if self.open_inventory:
            reward -= 1
        else:
            reward += 1
        self.previous_inventory_action = action["inventory"]
        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        self.episode_over = done
        return observation, reward, done, info

    def reset(self) -> dict:
        self.episode_over = False
        obs = super().reset()
        return obs


def _my_setting_gym_entrypoint(env_spec, fake=False):  # noqa: ANN001, ANN202
    """Used as entrypoint for `gym.make`."""
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)  # noqa: SLF001
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)  # noqa: SLF001

    env = MySettingWrapper(env)
    return env


OBTAIN_DIAMOND_SHOVEL_ENTRY_POINT = "my_env:_my_setting_gym_entrypoint"


class MySettingEnvSpec(HumanSurvival):
    def __init__(self) -> None:
        super().__init__(
            name="MineRLMySetting-v0",
            max_episode_steps=TIMEOUT,
            # Hardcoded variables to match the pretrained models
            fov_range=[70, 70],
            resolution=[640, 360],
            gamma_range=[2, 2],
            guiscale_range=[1, 1],
            cursor_size_range=[16.0, 16.0],
        )

    def _entry_point(self, fake: bool) -> str:  # noqa: ARG002
        return OBTAIN_DIAMOND_SHOVEL_ENTRY_POINT

    def create_observables(self) -> list[Handler]:
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(ALL_ITEMS),
        ]

    def create_monitors(self) -> list[TranslationHandler]:
        return []


my_setting_env_spec = MySettingEnvSpec()
my_setting_env_spec.register()

if __name__ == "__main__":
    from tqdm import tqdm

    env = gym.make("MineRLMySetting-v0")
    obs = env.reset()
    done = False
    step = 0

    MAX_STEP = 18000

    reward_list = []

    progress = tqdm(total=MAX_STEP)

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step += 1
        progress.update(1)
        env.render()
        if step == MAX_STEP:
            break

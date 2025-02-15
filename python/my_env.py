import gym
from minerl.env import _fake, _singleagent
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.mc import ALL_ITEMS

TIMEOUT = 18000


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

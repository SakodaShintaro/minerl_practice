"""Generate dataset with a random agent."""

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path

import gym
import minerl  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("save_root_dir", type=Path)
    parser.add_argument("--select_action_interval", type=int, default=10)
    parser.add_argument("--without_inventory_action", action="store_true", default=True)
    return parser.parse_args()


def fix_inv_dict(inv: OrderedDict) -> dict:
    result = {}
    for k, v in inv.items():
        if v.shape == ():
            if v > 0:
                result[k] = int(v)
        else:
            print(k, v, type(v), v.shape)
            raise AssertionError
    return result


if __name__ == "__main__":
    args = parse_args()
    save_root_dir = args.save_root_dir
    select_action_interval = args.select_action_interval

    logging.basicConfig(level=logging.DEBUG)

    save_root_dir.mkdir(exist_ok=True, parents=True)
    save_obs_dir = save_root_dir / "obs"
    save_obs_dir.mkdir(exist_ok=True)
    save_action_dir = save_root_dir / "action"
    save_action_dir.mkdir(exist_ok=True)
    save_inventory_dir = save_root_dir / "inventory"
    save_inventory_dir.mkdir(exist_ok=True)

    env = gym.make("MineRLObtainDiamondShovel-v0")
    obs = env.reset()
    inv = fix_inv_dict(obs["inventory"])
    obs = np.copy(obs["pov"])
    done = False
    step = 0

    MAX_STEP = 18000

    progress = tqdm(total=MAX_STEP)

    reward_list = []

    while not done:
        # Take a random action
        if step % select_action_interval == 0:
            action = env.action_space.sample()
            action["camera"][0] /= select_action_interval
            action["camera"][1] /= select_action_interval
            action["ESC"] = 0
            if args.without_inventory_action:
                action["inventory"] = 0

        # save obs and action
        Image.fromarray(obs).save(save_obs_dir / f"{step:08d}.png")
        with (save_action_dir / f"{step:08d}.json").open("w") as f:
            action_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in action.items()
            }
            json.dump(action_serializable, f)

        # save inventory
        with (save_inventory_dir / f"{step:08d}.json").open("w") as f:
            json.dump(inv, f)

        # step
        obs, reward, done, _ = env.step(action)
        inv = fix_inv_dict(obs["inventory"])
        obs = np.copy(obs["pov"])
        step += 1

        # save reward
        reward_list.append(reward)
        df = pd.DataFrame(reward_list)
        df.to_csv(save_root_dir / "reward.csv")

        env.render()

        progress.update(1)

        if step == MAX_STEP:
            break

    # 最後の結果を保存
    Image.fromarray(obs).save(save_obs_dir / f"{step:08d}.png")

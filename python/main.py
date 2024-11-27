"""Generate dataset with a random agent."""

import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

import gym
import minerl  # noqa: F401
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("save_root_dir", type=Path)
    parser.add_argument("--use_vae", action="store_true")
    parser.add_argument("--select_action_interval", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_root_dir = args.save_root_dir
    use_vae = args.use_vae
    select_action_interval = args.select_action_interval

    logging.basicConfig(level=logging.DEBUG)

    save_root_dir.mkdir(exist_ok=True, parents=True)
    save_obs_dir = save_root_dir / "obs"
    save_obs_dir.mkdir(exist_ok=True)
    save_action_dir = save_root_dir / "action"
    save_action_dir.mkdir(exist_ok=True)

    if use_vae:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        vae.eval()
        save_obs_hat_dir = save_root_dir / "obs_hat"
        save_obs_hat_dir.mkdir(exist_ok=True)

    env = gym.make("MineRLObtainDiamondShovel-v0")
    obs = env.reset()
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

        # save obs and action
        Image.fromarray(obs).save(save_obs_dir / f"{step:08d}.png")
        with (save_action_dir / f"{step:08d}.json").open("w") as f:
            action_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in action.items()
            }
            json.dump(action_serializable, f)

        # step
        obs, reward, done, _ = env.step(action)
        obs = np.copy(obs["pov"])
        step += 1

        # save reward
        reward_list.append(reward)
        df = pd.DataFrame(reward_list)
        df.to_csv(save_root_dir / "reward.csv")

        # vae
        if use_vae:
            x = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
            z = vae.encode(x).latent_dist.sample().mul_(0.18215)
            x_hat = vae.decode(z / 0.18215).sample
            obs_hat = x_hat.squeeze().permute(1, 2, 0).mul_(255).byte().cpu().numpy()
            Image.fromarray(obs_hat).save(save_obs_hat_dir / f"{step:08d}.png")

        env.render()

        progress.update(1)

        if step == MAX_STEP:
            break

    # 最後の結果を保存
    Image.fromarray(obs).save(save_obs_dir / f"{step:08d}.png")

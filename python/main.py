"""Test the environment with a random agent."""

import json
import logging
from pathlib import Path

import gym
import minerl  # noqa: F401
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image

logging.basicConfig(level=logging.DEBUG)
env = gym.make("MineRLObtainDiamondShovel-v0")
obs = env.reset()
obs = np.copy(obs["pov"])
done = False

CHECK_VAE_RECONSTRUCTION = False

if CHECK_VAE_RECONSTRUCTION:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()

save_root_dir = Path("minerl_dataset")
save_root_dir.mkdir(exist_ok=True)
save_obs_dir = save_root_dir / "obs"
save_obs_dir.mkdir(exist_ok=True)
save_action_dir = save_root_dir / "action"
save_action_dir.mkdir(exist_ok=True)
save_obs_hat_dir = save_root_dir / "obs_hat"
save_obs_hat_dir.mkdir(exist_ok=True)

step = 0

while not done:
    # Take a random action
    action = env.action_space.sample()
    action["ESC"] = 0
    print(action)

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

    # vae
    if CHECK_VAE_RECONSTRUCTION:
        x = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
        z = vae.encode(x).latent_dist.sample().mul_(0.18215)
        x_hat = vae.decode(z / 0.18215).sample
        obs_hat = x_hat.squeeze().permute(1, 2, 0).mul_(255).byte().cpu().numpy()
        Image.fromarray(obs_hat).save(save_obs_hat_dir / f"{step:08d}.png")

    env.render()

# 最後の結果を保存
Image.fromarray(obs).save(save_obs_dir / f"{step:08d}.png")

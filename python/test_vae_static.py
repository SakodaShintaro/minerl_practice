"""Test vae."""

import logging
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image

logging.basicConfig(level=logging.DEBUG)

device = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
vae.eval()

image_save_dir = Path("obs_images")
image_path_list = sorted(image_save_dir.glob("*.png"))
for image_path in image_path_list:
    obs = Image.open(image_path)
    obs = np.array(obs)
    x = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
    print(x.shape, x.dtype, x.min(), x.max())
    z = vae.encode(x).latent_dist.sample().mul_(0.18215)
    print(z.shape, z.dtype, z.min(), z.max())
    x_hat = vae.decode(z / 0.18215).sample
    print(x_hat.shape, x_hat.dtype, x_hat.min(), x_hat.max())

    img = x_hat.squeeze().permute(1, 2, 0).mul_(255).byte().cpu().numpy()
    img = Image.fromarray(img)
    img.save(image_path.with_name(f"recon_{image_path.name}"))
    break

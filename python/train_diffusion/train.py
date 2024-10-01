# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A minimal training script for DiT."""

import argparse
import logging
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from minerl_dataset import MineRLDataset
from models import DiT_models
from PIL import Image
from sample import sample_images
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# the first flag below was False when we tested this script but True makes training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, default="results")
    parser.add_argument("--global_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=5_000)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--image_size", type=int, default=128)
    return parser.parse_args()


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: torch.nn.Module, flag: bool) -> None:  # noqa: FBT001
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir: str) -> logging.Logger:
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def center_crop_arr(pil_image: Image, image_size: int) -> Image:
    """Center cropping implementation from ADM.

    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################


if __name__ == "__main__":
    """Trains a new DiT model."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    args = parse_args()

    device = 0
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting seed={seed}.")

    # Setup an experiment folder:
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(results_dir)
    logger.info(f"Experiment directory created at {results_dir}")

    # Create model:
    image_size = args.image_size
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size // 8
    ckpt = torch.load(args.ckpt) if args.ckpt is not None else None
    model = DiT_models[args.model](input_size=(latent_size, latent_size))
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    if ckpt is not None:
        ema.load_state_dict(ckpt["ema"])
    requires_grad(ema, flag=False)
    model = model.to(device)
    diffusion = create_diffusion(
        timestep_respacing="",
    )  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if ckpt is not None:
        opt.load_state_dict(ckpt["opt"])

    # Setup data
    dataset = MineRLDataset(args.data_path, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    limit_steps = args.steps
    train_steps = 0
    log_steps = 0
    running_loss = 0
    epoch = 0
    start_time = time()

    while True:
        epoch += 1
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            image, action = batch
            image = image.to(device)  # [b, seq, c, h, w]
            action = action.to(device)  # [b, seq, action_dim]
            b, seq, c, h, w = image.shape
            hidden_h = h // 8
            hidden_w = w // 8
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                image = image.view(b * seq, c, h, w)
                image = vae.encode(image).latent_dist.sample().mul_(0.18215)
                image = image.view(b, seq, 4, hidden_h, hidden_w)
            t = torch.randint(0, diffusion.num_timesteps, (image.shape[0],), device=device)

            cond_image = image[:, :-1]
            pred_image = image[:, -1:]

            model_kwargs = {"cond_image": cond_image, "cond_action": action}
            loss_dict = diffusion.training_losses(model, pred_image, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}",
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 or train_steps == 1:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                model.eval()
                samples = sample_images(model, vae, args)
                pred, gt, action = samples[0]
                save_image(
                    pred,
                    results_dir / f"{train_steps:08d}_pred.png",
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
                save_image(
                    gt,
                    results_dir / f"{train_steps:08d}_gt.png",
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
                model.train()

            if train_steps >= limit_steps:
                break
        if train_steps >= limit_steps:
            break

    # Save final checkpoint:
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
    }
    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()
    samples = sample_images(model, vae, args)
    pred, gt, action = samples[0]
    save_image(
        pred,
        results_dir / "last_pred.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(
        gt,
        results_dir / "last_gt.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )

    logger.info("Done!")

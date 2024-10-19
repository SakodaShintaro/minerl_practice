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

import torch
from diffusers.models import AutoencoderKL
from minerl_dataset import MineRLDataset
from models import DiT_models
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--nfe", type=int, default=100, help="Number of Function Evaluations")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--results_dir", type=Path, default="results")
    parser.add_argument("--steps", type=int, default=5_000)
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


def sample_images(
    loader: DataLoader,
    model: torch.nn.Module,
    vae: AutoencoderKL,
    args: argparse.Namespace,
) -> torch.Tensor:
    image_size = args.image_size
    sample_n = args.nfe
    with torch.no_grad():
        device = model.parameters().__next__().device
        latent_size = image_size // 8

        results = []

        for batch in loader:
            image, action = batch
            image = image.to(device)  # [b, seq, c, h, w]
            gt_image = image[:, -1]
            action = action.to(device)  # [b, seq, action_dim]
            b, seq, c, h, w = image.shape
            hidden_h = h // 8
            hidden_w = w // 8
            # Map input images to latent space + normalize latents:
            image = image.view(b * seq, c, h, w)
            image = vae.encode(image).latent_dist.sample().mul_(0.18215)
            image = image.view(b, seq, 4, hidden_h, hidden_w)

            cond_image = image[:, :-1]
            cond_action = action[:, :-1]

            # Create sampling noise:
            z = torch.randn(b, 1, 4, latent_size, latent_size, device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            cond_image = torch.cat([cond_image, cond_image], 0)
            cond_action = torch.cat([cond_action, cond_action], 0)

            dt = 1.0 / sample_n
            for i in range(sample_n):
                num_t = i / sample_n * (1 - eps) + eps
                t = torch.ones(b, device=device) * num_t
                t = torch.cat([t, t], 0)
                pred = model.forward(z, t * 999, cond_image, cond_action)
                cond, uncond = pred.chunk(2, 0)
                pred = uncond + (cond - uncond) * args.cfg_scale
                pred = torch.cat([pred, pred], 0)
                z = z.detach().clone() + pred * dt

            samples = z[:b, 0]
            pred_image = vae.decode(samples / 0.18215).sample
            results.append((pred_image, gt_image, action))
            break

        return results


def save_ckpt(  # noqa: PLR0913
    loader: DataLoader,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    train_steps: int,
) -> None:
    results_dir = args.results_dir
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/{train_steps:08d}.pt"
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
    }
    torch.save(checkpoint, checkpoint_path)
    model.eval()
    samples = sample_images(loader, model, vae, args)
    sample_dir = results_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    pred, gt, action = samples[0]
    save_image(
        pred,
        sample_dir / f"{train_steps:08d}_pred.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(
        gt,
        sample_dir / f"{train_steps:08d}_gt.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )


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
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{results_dir}/log.txt")],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment directory created at {results_dir}")

    # Create model:
    image_size = args.image_size
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size // 8
    ckpt = torch.load(args.ckpt) if args.ckpt is not None else None
    model = DiT_models[args.model](
        input_size=latent_size,
        learn_sigma=False,
    )
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    if ckpt is not None:
        ema.load_state_dict(ckpt["ema"])
    requires_grad(ema, flag=False)
    model = model.to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    if ckpt is not None:
        opt.load_state_dict(ckpt["opt"])

    # Setup data:
    train_dataset = MineRLDataset(args.data_path, image_size=image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(train_dataset)=:,} images ({args.data_path})")

    valid_dataset = MineRLDataset(args.data_path, image_size=image_size)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    log_every = args.steps // 100
    ckpt_every = args.steps // 10

    eps = 0.001

    save_ckpt(valid_loader, model, ema, opt, args, train_steps)
    model.train()

    logger.info(f"Training for {args.steps} steps...")
    for epoch in range(1, 100000):
        for image, action in train_loader:
            image = image.to(device)
            action = action.to(device)
            b, seq, c, h, w = image.shape
            hidden_h = h // 8
            hidden_w = w // 8
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                image = image.view(b * seq, c, h, w)
                image = vae.encode(image).latent_dist.sample().mul_(0.18215)
                image = image.view(b, seq, 4, hidden_h, hidden_w)

            cond_image = image[:, :-1]
            pred_image = image[:, -1:]
            cond_action = action[:, :-1]

            noise = torch.randn_like(pred_image)
            t = torch.rand(b, device=device) * (1 - eps) + eps
            t = t.view(-1, 1, 1, 1)
            perturbed_data = t * pred_image + (1 - t) * noise
            t = t.squeeze()
            out = model(perturbed_data, t * 999, cond_image, cond_action)
            target = pred_image - noise
            loss = torch.mean(torch.square(out - target))
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                logger.info(
                    f"(epoch={epoch:04d}) "
                    f"(step={train_steps:07d}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}",
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % ckpt_every == 0:
                save_ckpt(valid_loader, model, ema, opt, args, train_steps)
                model.train()

            if train_steps >= args.steps:
                break

        if train_steps >= args.steps:
            break

    # Save final checkpoint:
    save_ckpt(valid_loader, model, ema, opt, args, train_steps)
    logger.info("Done!")

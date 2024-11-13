"""Utility functions for training models."""

import argparse
from collections import OrderedDict
from shutil import rmtree

import torch
from diffusers.models import AutoencoderKL


def requires_grad(model: torch.nn.Module, flag: bool) -> None:  # noqa: FBT001
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def save_ckpt(
    model: torch.nn.Module,
    ema: torch.nn.Module,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    train_steps: int,
) -> None:
    results_dir = args.results_dir
    checkpoint_dir = results_dir / "checkpoints"
    rmtree(checkpoint_dir, ignore_errors=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/{train_steps:08d}.pt"
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
    }
    torch.save(checkpoint, checkpoint_path)


def second_to_str(seconds: float) -> str:
    """Convert seconds to a human-readable string."""
    second_int = int(seconds)
    minutes, seconds = divmod(second_int, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:03d}:{minutes:02d}:{seconds:02d}"


def sample_images_by_flow_matching(
    model: torch.nn.Module,
    feature: torch.Tensor,
    vae: AutoencoderKL,
    args: argparse.Namespace,
) -> torch.Tensor:
    image_size = args.image_size
    sample_n = args.nfe
    eps = 0.001
    b = 1
    with torch.no_grad():
        device = model.parameters().__next__().device
        latent_size = image_size // 8

        # Create sampling noise:
        z = torch.randn(b, 4, latent_size, latent_size, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)

        dt = 1.0 / sample_n
        for i in range(sample_n):
            num_t = i / sample_n * (1 - eps) + eps
            t = torch.ones(b, device=device) * num_t
            t = torch.cat([t, t], 0)
            pred = model.predict(z, t * 999, feature)
            cond, uncond = pred.chunk(2, 0)
            pred = uncond + (cond - uncond) * args.cfg_scale
            pred = torch.cat([pred, pred], 0)
            z = z.detach().clone() + pred * dt

        samples = z[:b]
        return vae.decode(samples / 0.18215).sample

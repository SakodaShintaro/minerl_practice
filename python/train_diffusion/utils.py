"""Utility functions for training models."""

import argparse
from collections import OrderedDict
from copy import deepcopy
from shutil import rmtree

import torch
from diffusers.models import AutoencoderKL
from models import DiT_models
from torchvision.utils import save_image


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


def create_models(
    args: argparse.Namespace,
    device: int,
) -> tuple[torch.nn.Module, torch.nn.Module, AutoencoderKL, torch.optim.Optimizer]:
    # Create model:
    image_size = args.image_size
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size // 8
    ckpt = torch.load(args.ckpt) if args.ckpt is not None else None
    model = DiT_models[args.model](
        input_size=(latent_size, latent_size),
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
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    if ckpt is not None:
        opt.load_state_dict(ckpt["opt"])
    return model, ema, vae, opt


def save_image_t(image: torch.Tensor, path: str) -> None:
    save_image(
        image,
        path,
        normalize=True,
        value_range=(-1, 1),
    )


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


def loss_flow_matching(
    model: torch.nn.Module,
    curr_image: torch.Tensor,
    feature: torch.Tensor,
) -> torch.Tensor:
    device = model.parameters().__next__().device
    noise = torch.randn_like(curr_image)
    eps = 0.001
    t = torch.rand(1, device=device) * (1 - eps) + eps
    t = t.view(-1, 1, 1, 1)
    perturbed_data = t * curr_image + (1 - t) * noise
    t = t.squeeze((1, 2, 3))
    out = model.predict(perturbed_data, t * 999, feature)
    target = curr_image - noise
    return torch.mean(torch.square(out - target))


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

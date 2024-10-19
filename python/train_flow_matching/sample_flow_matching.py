"""Sample by flow matching."""

import argparse

import torch
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader


def sample_images(
    loader: DataLoader,
    model: torch.nn.Module,
    vae: AutoencoderKL,
    args: argparse.Namespace,
) -> torch.Tensor:
    image_size = args.image_size
    sample_n = args.nfe
    eps = 0.001
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

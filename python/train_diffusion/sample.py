# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sample new images from a pre-trained DiT."""

import argparse
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from minerl_dataset import MineRLDataset
from models import DiT_models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def sample_images(model: torch.nn.Module, vae: AutoencoderKL) -> torch.Tensor:
    image_size = 256
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ],
    )
    dataset = MineRLDataset(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    device = model.parameters().__next__().device
    latent_size = image_size // 8

    results = []

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

        cond_image = image[:, :-1]
        pred_image = image[:, -1:]

        diffusion = create_diffusion(str(250))

        # Create sampling noise:
        z = torch.randn(b, 1, 4, latent_size, latent_size, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        cond_image = torch.cat([cond_image, cond_image], 0)
        action = torch.cat([action, action], 0)
        model_kwargs = {"cond_image": cond_image, "cond_action": action, "cfg_scale": 0.0}

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )
        print(f"{samples.shape=}")
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        print(f"{samples.shape=}")
        samples = samples[:, 0]
        print(f"{samples.shape=}")
        result = vae.decode(samples / 0.18215).sample
        results.append(result)
        return result

    return results


def main(args: argparse.Namespace) -> None:
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    num_classes = args.num_classes
    model = DiT_models[args.model](input_size=latent_size, num_classes=num_classes).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    state_dict = torch.load(str(args.ckpt))
    state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    samples = sample_images(model, vae)

    # Save and display images:
    save_dir = args.ckpt.parent.parent
    save_image(samples, save_dir / "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()
    main(args)

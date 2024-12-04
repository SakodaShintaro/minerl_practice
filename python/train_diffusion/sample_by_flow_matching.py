"""Sample by flow matching."""

import argparse
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from minerl_dataset import MineRLDataset
from models import DiT_models
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=Path, required=True)
    return parser.parse_args()


def sample_images_by_flow_matching(
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

            feature = model.extract_features(cond_image, cond_action)
            feature = torch.cat([feature, feature], 0)

            # Create sampling noise:
            z = torch.randn(b, 4, latent_size, latent_size, device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)

            dt = 1.0 / sample_n
            for i in range(sample_n):
                num_t = i / sample_n * (1 - eps) + eps
                t = torch.ones(b, device=device) * num_t
                t = torch.cat([t, t], 0)
                pred = model.predict(z, t, torch.zeros_like(t), feature)
                cond, uncond = pred.chunk(2, 0)
                pred = uncond + (cond - uncond) * args.cfg_scale
                pred = torch.cat([pred, pred], 0)
                z = z.detach().clone() + pred * dt

            samples = z[:b]
            pred_image = vae.decode(samples / 0.18215).sample
            results.append((pred_image, gt_image, action))
            break

        return results


if __name__ == "__main__":
    args = parse_args()

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    state_dict = torch.load(str(args.ckpt))
    train_args = state_dict["args"]
    args.image_size = train_args.image_size
    latent_size = train_args.image_size // 8
    model = DiT_models[train_args.model](
        input_size=latent_size,
        learn_sigma=not train_args.use_flow_matching,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    model.load_state_dict(state_dict["model"])
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    dataset = MineRLDataset(args.data_path, image_size=train_args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    args.nfe = train_args.nfe
    args.cfg_scale = train_args.cfg_scale
    result_list = sample_images_by_flow_matching(loader, model, vae, args)

    save_dir = args.ckpt.parent.parent / "samples"
    save_dir.mkdir(exist_ok=True, parents=True)

    for i, data_tuple in enumerate(result_list):
        pred, gt, action = data_tuple
        save_image(
            pred,
            save_dir / f"{i:08d}_sample_pred.png",
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
        )
        save_image(
            gt,
            save_dir / f"{i:08d}_sample_gt.png",
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
        )

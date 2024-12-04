# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from shutil import rmtree
from time import time

import pandas as pd
import torch
from minerl_dataset import MineRLDataset
from sample_by_flow_matching import sample_images_by_flow_matching
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import update_ema, second_to_str, create_models, loss_flow_matching

# the first flag below was False when we tested this script but True makes training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--model", type=str, default="DiT-S/2")
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=(16 + 1))
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    return parser.parse_args()


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
    model.eval()
    samples = sample_images_by_flow_matching(loader, model, vae, args)
    for i, sample in enumerate(samples):
        pred, gt, action = sample
        save_image(
            pred,
            results_dir / "predict" / f"{train_steps:08d}_{i:04d}.png",
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
        )
        save_image(
            gt,
            results_dir / "gt" / f"{train_steps:08d}_{i:04d}.png",
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
        )


if __name__ == "__main__":
    args = parse_args()

    device = 0
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting seed={seed}.")

    # Setup an experiment folder:
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    pr_save_dir = results_dir / "predict"
    pr_save_dir.mkdir(parents=True, exist_ok=True)
    gt_save_dir = results_dir / "gt"
    gt_save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{results_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)

    # Create model:
    model, ema, vae, opt = create_models(args, device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data
    dataset = MineRLDataset(
        args.data_path, image_size=args.image_size, seq_len=args.seq_len
    )
    logger.info(f"Train Dataset contains {len(dataset):,} images ({args.data_path})")

    train_loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Variables for monitoring/logging purposes:
    limit_steps = args.steps
    epoch = 0
    train_steps = 0
    train_loss_fm = 0
    train_loss_sc = 0
    valid_loss = 0
    ckpt_every = limit_steps // 10
    log_every = max(limit_steps // 200, 1)
    logger.info(f"{ckpt_every=}, {log_every=}")

    start_time = time()

    save_ckpt(valid_loader, model, ema, opt, args, train_steps)
    model.train()

    log_dict_list = []

    while True:
        epoch += 1
        logger.info(f"Beginning epoch {epoch}...")
        for batch in train_loader:
            image, action = batch
            image = image.to(device)  # [b, seq, c, h, w]
            action = action.to(device)  # [b, seq, action_dim]
            b, seq, c, h, w = image.shape
            hidden_h = h // 8
            hidden_w = w // 8
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                latent_image = image.view(b * seq, c, h, w)
                latent_image = (
                    vae.encode(latent_image).latent_dist.sample().mul_(0.18215)
                )
                latent_image = latent_image.view(b, seq, 4, hidden_h, hidden_w)

            cond_image = latent_image[:, :-1]
            pred_image = latent_image[:, -1]
            cond_action = action[:, :-1]

            # flow matchingの学習
            feature = model.extract_features(cond_image, cond_action)
            loss_fm, loss_sc = loss_flow_matching(model, pred_image, feature)
            loss = loss_fm + 0.0 * loss_sc
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            train_loss_fm += loss_fm.item()
            train_loss_sc += loss_sc.item()
            train_steps += 1
            if train_steps % log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                elapsed_time = end_time - start_time
                remaining_time = (
                    elapsed_time * (limit_steps - train_steps) / train_steps
                )
                elapsed_time_str = second_to_str(elapsed_time)
                remaining_time_str = second_to_str(remaining_time)

                train_loss_fm /= log_every
                train_loss_sc /= log_every
                valid_loss /= log_every
                logger.info(
                    f"remaining_time={remaining_time_str} "
                    f"elapsed_time={elapsed_time_str} "
                    f"epoch={epoch:04d} "
                    f"step={train_steps:08d} "
                    f"loss_fm={train_loss_fm:.4f} "
                    f"loss_sc={train_loss_sc:.4f} "
                    f"loss_image={valid_loss:.4f}",
                )
                log_dict_list.append(
                    {
                        "elapsed_time": elapsed_time_str,
                        "epoch": epoch,
                        "step": train_steps,
                        "loss_fm": train_loss_fm,
                        "loss_sc": train_loss_sc,
                        "loss_image": valid_loss,
                    },
                )
                df = pd.DataFrame(log_dict_list)
                df.to_csv(results_dir / "log.tsv", index=False, sep="\t")
                # Reset monitoring variables:
                train_loss_fm = 0
                train_loss_sc = 0
                valid_loss = 0

            # Save DiT checkpoint:
            if train_steps % ckpt_every == 0:
                save_ckpt(valid_loader, model, ema, opt, args, train_steps)
                model.train()

            if train_steps >= limit_steps:
                break
        if train_steps >= limit_steps:
            break

    # Save final checkpoint:
    save_ckpt(valid_loader, model, ema, opt, args, train_steps)
    logger.info("Done!")

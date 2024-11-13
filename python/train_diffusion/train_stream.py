# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A minimal training script for DiT."""

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from time import time

import pandas as pd
import torch
from diffusers.models import AutoencoderKL
from minerl_dataset import MineRLDataset
from models import DiT_models
from torch.utils.data import DataLoader
from utils import (
    loss_flow_matching,
    requires_grad,
    sample_images_by_flow_matching,
    save_ckpt,
    save_image_t,
    second_to_str,
    update_ema,
)

# the first flag below was False when we tested this script but True makes training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--nfe", type=int, default=100, help="Number of Function Evaluations")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--results_dir", type=Path, default="results")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    return parser.parse_args()


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
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pr_save_dir = results_dir / "predict"
    pr_save_dir.mkdir(parents=True, exist_ok=True)
    gt_save_dir = results_dir / "gt"
    gt_save_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
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
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()
    logger.info("Finish setup ema")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    if ckpt is not None:
        opt.load_state_dict(ckpt["opt"])

    # Setup data
    dataset = MineRLDataset(args.data_path, image_size=image_size, seq_len=1)
    logger.info(f"Train Dataset contains {len(dataset):,} images ({args.data_path})")

    train_loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Variables for monitoring/logging purposes:
    limit_steps = args.steps
    train_steps = 0
    log_steps = 0
    running_loss = 0
    epoch = 0
    start_time = time()
    loss_image_ave = 0
    validate_period = 1000  # 1000ステップごとに
    validate_num = 10  # 10ステップ出力する
    ckpt_every = args.steps // 10
    log_every = max(args.steps // 200, 1)
    logger.info(f"{ckpt_every=}, {log_every=}")

    save_ckpt(model, ema, opt, args, train_steps)
    model.train()

    feature, conv_state, ssm_state = model.allocate_inference_cache(args.batch_size)

    log_dict_list = []

    while True:
        epoch += 1
        logger.info(f"Beginning epoch {epoch}...")
        for batch in train_loader:
            # (1) 画像tが得られる
            image, action = batch
            image = image.to(device)  # [1, 1, c, h, w]
            action = action.to(device)  # [1, 1, action_dim]
            image_gt = image[:, 0]
            b, seq, c, h, w = image.shape
            hidden_h = h // 8
            hidden_w = w // 8
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                image = image.view(b * seq, c, h, w)
                image = vae.encode(image).latent_dist.sample().mul_(0.18215)
                image = image.view(b, seq, 4, hidden_h, hidden_w)

            curr_image = image[0]  # [1, 4, hidden_h, hidden_w]
            curr_action = action[0]  # [1, action_dim]

            # validate
            if train_steps % validate_period < validate_num:
                if train_steps % validate_period == 0:
                    loss_image_ave = 0
                pred_image = sample_images_by_flow_matching(model, feature, vae, args)
                save_image_t(pred_image, pr_save_dir / f"{train_steps:08d}.png")
                save_image_t(image_gt, gt_save_dir / f"{train_steps:08d}.png")
                diff = pred_image - image_gt
                loss_image = torch.mean(torch.square(diff))
                loss_image_ave += loss_image.item() / validate_num

            # (2) flow matchingの学習
            loss = loss_flow_matching(model, curr_image, feature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # (3) Update state
            feature, conv_state, ssm_state = model.step(
                curr_image,
                curr_action,
                conv_state,
                ssm_state,
            )

            log_steps += 1
            train_steps += 1

            if train_steps == 1:
                continue

            # Log loss values:
            running_loss += loss.item()
            if train_steps % log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                elapsed_time = end_time - start_time
                remaining_time = elapsed_time * (limit_steps - train_steps) / train_steps
                elapsed_time_str = second_to_str(elapsed_time)
                remaining_time_str = second_to_str(remaining_time)

                avg_loss = running_loss / log_steps
                logger.info(
                    f"remaining_time={remaining_time_str} "
                    f"elapsed_time={elapsed_time_str} "
                    f"epoch={epoch:04d} "
                    f"step={train_steps:08d} "
                    f"loss={avg_loss:.4f} "
                    f"loss_image={loss_image_ave:.4f}",
                )
                log_dict_list.append(
                    {
                        "elapsed_time": elapsed_time,
                        "epoch": epoch,
                        "step": train_steps,
                        "loss": avg_loss,
                        "loss_imag": loss_image_ave,
                    },
                )
                df = pd.DataFrame(log_dict_list)
                df.to_csv(results_dir / "log.tsv", index=False, sep="\t")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0

            # Save DiT checkpoint:
            if train_steps % ckpt_every == 0:
                save_ckpt(model, ema, opt, args, train_steps)
                model.train()

            if train_steps >= limit_steps:
                break
        if train_steps >= limit_steps:
            break

    # Save final checkpoint:
    save_ckpt(model, ema, opt, args, train_steps)
    logger.info("Done!")

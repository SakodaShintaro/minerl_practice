# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A minimal training script for DiT."""

import argparse
import logging
from pathlib import Path
from time import time

import pandas as pd
import torch
from minerl_dataset import MineRLDataset, action_tensor_to_dict
from models import DiT_models
from torch.utils.data import DataLoader
from utils import (
    create_models,
    loss_flow_matching,
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
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2"
    )
    parser.add_argument(
        "--nfe", type=int, default=100, help="Number of Function Evaluations"
    )
    parser.add_argument("--results_dir", type=Path, default="results")
    parser.add_argument("--limit_steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    pr_save_dir = results_dir / "predict"
    pr_save_dir.mkdir(parents=True, exist_ok=True)
    gt_save_dir = results_dir / "gt"
    gt_save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
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
    dataset = MineRLDataset(args.data_path, image_size=args.image_size, seq_len=1)
    logger.info(f"Train Dataset contains {len(dataset):,} images ({args.data_path})")

    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    # Variables for monitoring/logging purposes:
    limit_steps = args.limit_steps
    epoch = 0
    train_steps = 0
    train_loss_fm = 0
    train_loss_sc = 0
    valid_loss = 0
    VALIDATE_EVERY = 1000  # 1000ステップごとに
    VALIDATE_NUM = 10  # 10ステップ出力する
    ckpt_every = limit_steps // 10
    log_every = max(limit_steps // 200, 1)
    logger.info(f"{ckpt_every=}, {log_every=}")

    start_time = time()

    save_ckpt(model, ema, opt, args, train_steps)
    model.train()

    feature, conv_state, ssm_state = model.allocate_inference_cache(batch_size=1)

    log_dict_list = []

    while True:
        epoch += 1
        logger.info(f"Beginning epoch {epoch}...")
        for batch in train_loader:
            train_steps += 1

            # current inference
            # (1) action
            action, log_prob, entropy = model.policy(feature)
            action_dict = action_tensor_to_dict(action)

            # (2) value
            curr_value = model.value(feature)

            # env step
            image, action = batch
            image, action = image[:, 0], action[:, 0]
            image = image.to(device)  # [1, c, h, w]
            action = action.to(device)  # [1, action_dim]
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                # The shape is [1, 4, h // 8, w // 8]
                latent_image = vae.encode(image).latent_dist.sample().mul_(0.18215)

            # validate
            if train_steps % VALIDATE_EVERY < VALIDATE_NUM:
                if train_steps % VALIDATE_EVERY == 0:
                    valid_loss = 0
                pred_image = sample_images_by_flow_matching(model, feature, vae, args)
                save_image_t(pred_image, pr_save_dir / f"{train_steps:08d}.png")
                save_image_t(image, gt_save_dir / f"{train_steps:08d}.png")
                diff = pred_image - image
                loss_image = torch.mean(torch.square(diff))
                valid_loss += loss_image.item() / VALIDATE_NUM

            # flow matchingの学習
            loss_fm, loss_sc = loss_flow_matching(model, latent_image, feature)
            loss = loss_fm + loss_sc
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Update state
            feature, conv_state, ssm_state = model.step(
                latent_image,
                action,
                conv_state,
                ssm_state,
            )

            if train_steps == 1:
                continue

            # Log loss values:
            train_loss_fm += loss_fm.item()
            train_loss_sc += loss_sc.item()
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
                        "elapsed_time": elapsed_time,
                        "epoch": epoch,
                        "step": train_steps,
                        "loss_fm": train_loss_fm,
                        "loss_sc": train_loss_sc,
                        "loss_imag": valid_loss,
                    },
                )
                df = pd.DataFrame(log_dict_list)
                df.to_csv(results_dir / "log.tsv", index=False, sep="\t")
                train_loss_fm = 0
                train_loss_sc = 0

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

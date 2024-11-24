# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A minimal training script for DiT."""

import argparse
from pathlib import Path
from time import time
from tqdm import tqdm

import torch
from minerl_dataset import MineRLDataset, action_tensor_to_dict
from torch.utils.data import DataLoader
from utils import (
    create_models,
    sample_images_by_flow_matching,
    save_image_t,
)

# the first flag below was False when we tested this script but True makes training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=Path)
    parser.add_argument("--nfe", type=int, default=100, help="Number of Function Evaluations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt
    nfe = args.nfe

    device = 0
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting seed={seed}.")

    ckpt = torch.load(ckpt_path)
    args = ckpt["args"]
    args.ckpt = ckpt_path
    args.nfe = nfe
    args.results_dir = ckpt_path.parent.parent

    print(args)

    # Setup an experiment folder:
    results_dir = args.results_dir
    assert results_dir.exists(), f"{results_dir} does not exist."
    pr_save_dir = results_dir / "eval_predict"
    pr_save_dir.mkdir(parents=True, exist_ok=True)
    gt_save_dir = results_dir / "eval_gt"
    gt_save_dir.mkdir(parents=True, exist_ok=True)

    # Create model:
    model, ema, vae, _ = create_models(args, device)

    # Setup data
    dataset = MineRLDataset(args.data_path, image_size=args.image_size, seq_len=1)

    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    # Variables for monitoring/logging purposes:
    limit_steps = 40
    valid_steps = 0
    valid_loss = 0

    start_time = time()

    feature, conv_state, ssm_state = model.allocate_inference_cache(batch_size=1)

    log_dict_list = []

    progress = tqdm(total=limit_steps)

    with torch.no_grad():
        for batch in train_loader:
            valid_steps += 1

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
            pred_image = sample_images_by_flow_matching(model, feature, vae, args)
            save_image_t(pred_image, pr_save_dir / f"{valid_steps:08d}.png")
            save_image_t(image, gt_save_dir / f"{valid_steps:08d}.png")
            diff = pred_image - image
            loss_image = torch.mean(torch.square(diff))
            valid_loss += loss_image.item() / limit_steps

            # Update state
            feature, conv_state, ssm_state = model.step(
                latent_image,
                action,
                conv_state,
                ssm_state,
            )

            progress.update(1)

            if valid_steps >= limit_steps:
                break

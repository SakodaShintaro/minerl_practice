"""Generate dataset with a random agent."""

import argparse
import logging
from pathlib import Path
from time import time

import gym
import minerl  # noqa: F401
import pandas as pd
import torch
from minerl_dataset import action_dict_to_tensor, action_tensor_to_dict
from mock_env import MockMineRL
from models import DiT_models
from PIL import Image
from torchvision import transforms
from utils import (
    create_models,
    loss_flow_matching,
    sample_images_by_flow_matching,
    save_ckpt,
    save_image_t,
    second_to_str,
    update_ema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--nfe", type=int, default=100, help="Number of Function Evaluations")
    parser.add_argument("--results_dir", type=Path, default="../../train_result/stream_rl")
    parser.add_argument("--limit_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_mock", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set seed
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
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{results_dir}/log.txt")],
    )
    logger = logging.getLogger(__name__)

    # Create model:
    model, ema, vae, opt = create_models(args, device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Variables for monitoring/logging purposes:
    limit_steps = args.limit_steps
    epoch = 0
    train_steps = 0
    train_loss = 0
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

    env = MockMineRL() if args.use_mock else gym.make("MineRLObtainDiamondShovel-v0")
    env.reset()
    done = False
    GAMMA = 0.9

    image_size = args.image_size
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ],
    )

    with torch.no_grad():
        eligibility_traces = [torch.zeros_like(p, requires_grad=False) for p in model.parameters()]
    torch.autograd.set_detect_anomaly(True)
    while True:
        env.reset()
        done = False
        epoch += 1
        logger.info(f"Beginning epoch {epoch}...")
        episode_steps = 0
        while not done:
            episode_steps += 1
            train_steps += 1

            # current inference
            # (1) action
            action, log_prob, entropy = model.policy(feature)
            action_dict = action_tensor_to_dict(action)

            # (2) value
            curr_value = model.value(feature)

            # env step
            obs, reward, done, _ = env.step(action_dict)
            env.render()
            obs = transform(Image.fromarray(obs["pov"]))
            image = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                # The shape is [1, 4, h // 8, w // 8]
                latent_image = vae.encode(image).latent_dist.sample().mul_(0.18215)

            # compare predict image and actual image
            if train_steps % VALIDATE_EVERY < VALIDATE_NUM:
                if train_steps % VALIDATE_EVERY == 0:
                    valid_loss = 0
                pred_image = sample_images_by_flow_matching(model, feature, vae, args)
                save_image_t(pred_image, pr_save_dir / f"{train_steps:08d}.png")
                save_image_t(image, gt_save_dir / f"{train_steps:08d}.png")
                diff = pred_image - image
                loss_image = torch.mean(torch.square(diff))
                valid_loss += loss_image.item() / VALIDATE_NUM

            # flow matching
            loss_f = loss_flow_matching(model, latent_image, feature)
            opt.zero_grad()
            loss_f.backward()
            opt.step()
            update_ema(ema, model)

            # model step
            curr_action = action_dict_to_tensor(action_dict).unsqueeze(0).to(device)
            feature, conv_state, ssm_state = model.step(
                latent_image,
                curr_action,
                conv_state,
                ssm_state,
            )

            # calculate temporal difference
            with torch.no_grad():
                next_value = model.value(feature)
                target = (reward + GAMMA * next_value)
                delta = target - curr_value

            # loss_v = curr_value.mean()
            # loss_p = (-log_prob).mean()
            # loss = loss_v + loss_p

            # update eligibility traces
            # opt.zero_grad()
            # loss.backward(retain_graph=True)
            # with torch.no_grad():
            #     for p, e in zip(model.parameters(), eligibility_traces):
            #         if p.grad is None:
            #             continue
            #         e.mul_(GAMMA).add_(p.grad)

            # backward f
            # opt.zero_grad()
            # loss_f.backward()

            # add eligibility traces
            # with torch.no_grad():
            #     for p, e in zip(model.parameters(), eligibility_traces):
            #         if p.grad is None:
            #             continue
            #         p.grad.add_(args.lr * delta.item() * e)

            # update
            # opt.step()

            train_loss += loss_f.item()
            if train_steps % log_every == 0:
                # Measure training speed:
                end_time = time()
                elapsed_time = end_time - start_time
                remaining_time = elapsed_time * (limit_steps - train_steps) / train_steps
                elapsed_time_str = second_to_str(elapsed_time)
                remaining_time_str = second_to_str(remaining_time)

                avg_loss = train_loss / log_every
                logger.info(
                    f"remaining_time={remaining_time_str} "
                    f"elapsed_time={elapsed_time_str} "
                    f"epoch={epoch:04d} "
                    f"step={train_steps:08d} "
                    f"loss={avg_loss:.4f} "
                    f"loss_image={valid_loss:.4f}",
                )
                log_dict_list.append(
                    {
                        "elapsed_time": elapsed_time,
                        "epoch": epoch,
                        "step": train_steps,
                        "loss": avg_loss,
                        "loss_imag": valid_loss,
                    },
                )
                df = pd.DataFrame(log_dict_list)
                df.to_csv(results_dir / "log.tsv", index=False, sep="\t")
                train_loss = 0

            # Save DiT checkpoint:
            if train_steps % ckpt_every == 0:
                save_ckpt(model, ema, opt, args, train_steps)
                model.train()

            if train_steps >= limit_steps:
                break

        if train_steps >= limit_steps:
            break
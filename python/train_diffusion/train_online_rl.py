import argparse
import logging
from collections import OrderedDict
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
    parser.add_argument("--nfe", type=int, default=4, help="Number of Function Evaluations")
    parser.add_argument("--results_dir", type=Path, default="../../train_result/stream_rl")
    parser.add_argument("--limit_steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_mock", action="store_true")
    parser.add_argument("--use_et", action="store_true")
    parser.add_argument("--select_action_interval", type=int, default=40)
    return parser.parse_args()


def fix_inv_dict(inv: OrderedDict) -> dict:
    result = {}
    total_num = 0
    for k, v in inv.items():
        if v.shape == ():
            if v > 0:
                result[k] = int(v)
                total_num += int(v)
        else:
            print(k, v, type(v), v.shape)
            raise AssertionError()
    return result, total_num


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
    train_loss_fm = 0
    train_loss_sc = 0
    valid_loss = 0
    VALIDATE_EVERY = 1000  # 1000ステップごとに
    VALIDATE_NUM = 10  # 10ステップ出力する
    ckpt_every = limit_steps // 10
    log_every = max(limit_steps // 200, 1)
    logger.info(f"{ckpt_every=}, {log_every=}")
    sum_reward = 0

    start_time = time()

    save_ckpt(model, ema, opt, args, train_steps)
    model.train()

    feature, conv_state, ssm_state = model.allocate_inference_cache(batch_size=1)

    log_dict_list = []

    env = MockMineRL() if args.use_mock else gym.make("MineRLObtainDiamondShovel-v0")
    obs = env.reset()
    obs_inv, obs_total_num = fix_inv_dict(obs["inventory"])
    done = False
    action = None
    GAMMA = 0.9

    image_size = args.image_size
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ],
    )

    if args.use_et:
        with torch.no_grad():
            eligibility_traces = [
                torch.zeros_like(p, requires_grad=False) for p in model.parameters()
            ]

    while True:
        env.reset()
        done = False
        epoch += 1
        logger.info(f"Beginning epoch {epoch}...")
        episode_steps = 0
        while not done:
            # current inference
            # (1) action
            if train_steps % args.select_action_interval == 0:
                # select
                action, log_prob, entropy = model.policy(feature)
                action_dict = action_tensor_to_dict(action)
            else:
                # recalculate only log_prob and entropy
                log_prob, entropy = model.evaluate(feature, action.detach())

            # (2) value
            curr_value = model.value(feature)

            # env step
            obs, reward, done, _ = env.step(action_dict)
            env.render()
            curr_obs_inv, curr_obs_total_num = fix_inv_dict(obs["inventory"])
            reward = curr_obs_total_num - obs_total_num
            obs_inv = curr_obs_inv
            obs_total_num = curr_obs_total_num
            obs_pov = transform(Image.fromarray(obs["pov"]))
            image = obs_pov.unsqueeze(0).to(device)
            sum_reward += reward
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                # The shape is [1, 4, h // 8, w // 8]
                latent_image = vae.encode(image).latent_dist.sample().mul_(0.18215)

            episode_steps += 1
            train_steps += 1

            # compare predict image and actual image
            pred_image = sample_images_by_flow_matching(model, feature, vae, args)
            diff = pred_image - image
            loss_image = torch.mean(torch.square(diff))
            valid_loss += loss_image.item()
            if train_steps % VALIDATE_EVERY < VALIDATE_NUM:
                save_image_t(pred_image, pr_save_dir / f"{train_steps:08d}.png")
                save_image_t(image, gt_save_dir / f"{train_steps:08d}.png")

            # flow matching
            loss_fm, loss_sc = loss_flow_matching(model, latent_image, feature)
            loss_wm = loss_fm + loss_sc
            opt.zero_grad()
            if args.use_et:
                loss_wm.backward(retain_graph=True)
                with torch.no_grad():
                    tmp_grad = [p.grad for p in model.parameters()]
            else:
                loss_wm.backward()
                opt.step()
                update_ema(ema, model)

            # model step
            curr_action = action_dict_to_tensor(action_dict).unsqueeze(0).to(device)
            new_feature, conv_state, ssm_state = model.step(
                latent_image,
                curr_action,
                conv_state,
                ssm_state,
            )

            # calculate temporal difference
            with torch.no_grad():
                next_value = model.value(feature)
                target = reward + GAMMA * next_value
                delta = target - curr_value

            # update eligibility traces
            if args.use_et:
                loss_v = curr_value.mean()
                loss_p = (-log_prob).mean()
                loss = 0.001 * (loss_v + loss_p)

                opt.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for p, e, g in zip(model.parameters(), eligibility_traces, tmp_grad):
                        if p.grad is None:
                            continue
                        e.mul_(GAMMA).add_(p.grad)
                        update = delta.item() * e
                        if g is not None:
                            update += g
                        p.data -= args.lr * update

                # update
                update_ema(ema, model)

            feature = new_feature[:, 0]

            train_loss_fm += loss_fm.item()
            train_loss_sc += loss_sc.item()
            if train_steps % log_every == 0:
                # Measure training speed:
                end_time = time()
                elapsed_time = end_time - start_time
                remaining_time = elapsed_time * (limit_steps - train_steps) / train_steps
                elapsed_time_str = second_to_str(elapsed_time)
                remaining_time_str = second_to_str(remaining_time)

                train_loss_fm /= log_every
                train_loss_sc /= log_every
                valid_loss /= log_every
                ave_reward = sum_reward / log_every
                logger.info(
                    f"remaining_time={remaining_time_str} "
                    f"elapsed_time={elapsed_time_str} "
                    f"epoch={epoch:04d} "
                    f"step={train_steps:08d} "
                    f"loss_fm={train_loss_fm:.4f} "
                    f"loss_sc={train_loss_sc:.4f} "
                    f"loss_image={valid_loss:.4f} "
                    f"ave_reward={ave_reward:.4f}",
                )
                log_dict_list.append(
                    {
                        "elapsed_time": elapsed_time,
                        "epoch": epoch,
                        "step": train_steps,
                        "loss_fm": train_loss_fm,
                        "loss_sc": train_loss_sc,
                        "loss_image": valid_loss,
                        "ave_reward": ave_reward,
                    },
                )
                df = pd.DataFrame(log_dict_list)
                df.to_csv(results_dir / "log.tsv", index=False, sep="\t")
                train_loss_fm = 0
                train_loss_sc = 0
                valid_loss = 0
                sum_reward = 0

            # Save DiT checkpoint:
            if train_steps % ckpt_every == 0:
                save_ckpt(model, ema, opt, args, train_steps)
                model.train()

            if train_steps >= limit_steps:
                break

        if train_steps >= limit_steps:
            break

"""Based on https://github.com/gauthamvasan/avg/blob/main/avg.py.

Copyright (c) [2024] [Gautham Vasan] - MIT License.
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import gym
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from torchvision import transforms

import my_env
import wandb
from network import Actor, SoftQNetwork
from reward_processor import RewardProcessor
from td_error_scaler import TDErrorScaler
from train_diffusion.mock_env import MockMineRL  # noqa: F401

IMAGE_H = 360 // 5
IMAGE_W = 640 // 5
Z_H = IMAGE_H // 8
H_W = IMAGE_W // 8
Z_CH = 4
INPUT_DIM = Z_H * H_W * Z_CH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--N", default=2_000_000, type=int)
    parser.add_argument("--actor_lr", default=0.00063, type=float)
    parser.add_argument("--critic_lr", default=0.00087, type=float)
    parser.add_argument("--alpha_lr", default=1e-2, type=float)
    parser.add_argument("--beta1", default=0.0, type=float)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--l2_actor", default=0.0, type=float)
    parser.add_argument("--l2_critic", default=0.0, type=float)
    parser.add_argument("--hidden_actor", default=256, type=int)
    parser.add_argument("--hidden_critic", default=2048, type=int)
    parser.add_argument("--use_eligibility_trace", action="store_true")
    parser.add_argument("--et_lambda", default=0.0, type=float)
    parser.add_argument("--reward_processing_type", default="none", type=str)
    parser.add_argument("--save_dir", default="./results", type=Path)
    parser.add_argument("--save_suffix", default="AVG", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--print_interval_episode", default=50, type=int)
    parser.add_argument("--without_entorpy_term", action="store_true")
    return parser.parse_args()


class AVG:
    """AVG Agent."""

    def __init__(self, cfg: argparse.Namespace) -> None:
        self.steps = 0

        self.actor = Actor(
            input_dim=INPUT_DIM,
            hidden_dim=256,
            action_dim=my_env.ACTION_DIM,
            use_normalize=False,
        ).to(cfg.device)
        self.Q = SoftQNetwork(
            input_dim=INPUT_DIM,
            action_dim=my_env.ACTION_DIM,
            hidden_dim=256,
            use_normalize=False,
        ).to(cfg.device)

        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr

        self.popt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            betas=cfg.betas,
            weight_decay=cfg.l2_actor,
        )
        self.qopt = torch.optim.AdamW(
            self.Q.parameters(),
            lr=cfg.critic_lr,
            betas=cfg.betas,
            weight_decay=cfg.l2_critic,
        )

        self.gamma, self.device = cfg.gamma, cfg.device
        self.td_error_scaler = TDErrorScaler()
        self.G = 0

        self.use_eligibility_trace = cfg.use_eligibility_trace

        self.et_lambda = cfg.et_lambda
        with torch.no_grad():
            self.eligibility_traces_q = [
                torch.zeros_like(p, requires_grad=False) for p in self.Q.parameters()
            ]

        if cfg.without_entorpy_term:
            self.log_alpha = None
        else:
            self.target_entropy = -my_env.ACTION_DIM
            self.log_alpha = torch.nn.Parameter(
                torch.zeros(1, requires_grad=True, device=cfg.device),
            )
            self.aopt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    def update(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
        lprob: torch.Tensor,
    ) -> None:
        """Update the actor and critic networks based on the observed transition."""
        #### Q loss
        q = self.Q(obs, action.detach())  # N.B: Gradient should NOT pass through action here
        with torch.no_grad():
            alpha = self.log_alpha.exp().item() if self.log_alpha is not None else 0.0
            next_action, next_lprob, mean = self.actor.get_action(next_obs)
            q2 = self.Q(next_obs, next_action)
            target_value = q2 - alpha * next_lprob

        #### Return scaling
        r_ent = reward - alpha * lprob.detach().item()
        self.G += r_ent
        if done:
            self.td_error_scaler.update(reward=r_ent, gamma=0, G=self.G)
            self.G = 0
        else:
            self.td_error_scaler.update(reward=r_ent, gamma=self.gamma, G=None)

        delta = reward + (1 - done) * self.gamma * target_value - q
        delta /= self.td_error_scaler.sigma

        # Policy loss
        ploss = alpha * lprob - self.Q(obs, action)  # N.B: USE reparametrized action
        self.popt.zero_grad()
        ploss.backward()
        self.popt.step()

        self.qopt.zero_grad()
        if self.use_eligibility_trace:
            q.backward()
            with torch.no_grad():
                for p, et in zip(self.Q.parameters(), self.eligibility_traces_q):
                    et.mul_(self.et_lambda * self.gamma).add_(p.grad.data)
                    p.grad.data = -2.0 * delta.item() * et
        else:
            qloss = delta**2
            qloss.backward()
        self.qopt.step()

        # alpha
        if self.log_alpha is None:
            alpha_loss = torch.Tensor([0.0])
        else:
            alpha_loss = (-self.log_alpha.exp() * (lprob.detach() + self.target_entropy)).mean()
            self.aopt.zero_grad()
            alpha_loss.backward()
            self.aopt.step()

        self.steps += 1

        return {
            "delta": delta.item(),
            "q": q.item(),
            "policy_loss": ploss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha,
        }

    def reset_eligibility_traces(self) -> None:
        """Reset eligibility traces."""
        for et in self.eligibility_traces_q:
            et.zero_()


if __name__ == "__main__":
    args = parse_args()

    # init wandb
    wandb.init(project="MineRL", name=args.save_suffix, config=args)

    # Adam
    args.betas = [args.beta1, 0.999]

    # CPU/GPU use for the run
    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = args.save_dir / f"{datetime_str}_{args.save_suffix}"
    save_dir.mkdir(exist_ok=True, parents=True)

    # Start experiment
    # N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow.
    # Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    torch.cuda.set_device(args.gpu_id)

    tic = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{save_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"AVG-seed-{args.seed}")

    logger.info("Command line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Env
    # env = MockMineRL()  # noqa: ERA001
    env = gym.make("MineRLMySetting-v0")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(args.device)

    #### Reproducibility
    env.reset()
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ####

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ],
    )

    # Learner
    agent = AVG(args)

    # Interaction
    reward_processor = RewardProcessor(args.reward_processing_type)
    sum_reward = 0
    sum_delta, sum_lprob = 0, 0
    sum_reward_normed = 0
    terminated = False
    obs = env.reset()
    obs = obs["pov"]
    obs = cv2.resize(obs, (IMAGE_W, IMAGE_H))
    data_list = []

    for total_step in range(1, args.N + 1):
        # N.B: Action is a torch.Tensor
        obs_tensor = transform(obs)
        obs_tensor = obs_tensor.unsqueeze(0).to(args.device)
        with torch.no_grad():
            latent_image = vae.encode(obs_tensor).latent_dist.sample().mul_(0.18215)
            latent = latent_image.flatten(start_dim=1)

        action, lprob, mean = agent.actor.get_action(latent)
        array_action = action.detach().cpu().view(-1).numpy()
        dict_action = my_env.array_action_to_dict_action(array_action)

        # Receive reward and next state
        next_obs, reward, termination, info = env.step(dict_action)
        env.render()
        next_obs = next_obs["pov"]
        next_obs = cv2.resize(next_obs, (IMAGE_W, IMAGE_H))
        with torch.no_grad():
            next_obs_tensor = transform(next_obs)
            next_obs_tensor = next_obs_tensor.unsqueeze(0).to(args.device)
            next_latent_image = vae.encode(next_obs_tensor).latent_dist.sample().mul_(0.18215)
            next_latent = next_latent_image.flatten(start_dim=1)

        reward_normed = reward_processor.normalize(reward)
        stats = agent.update(latent, action, next_latent, reward_normed, terminated, lprob)
        sum_delta += stats["delta"]
        sum_lprob += lprob.item()
        sum_reward += reward
        sum_reward_normed += reward_normed

        obs = next_obs

        if total_step % 1 == 0:
            step_data = {
                "global_step": total_step,
                "reward": reward,
                "losses/actor_loss": stats["policy_loss"],
                "losses/qf1_values": stats["q"],
                "losses/alpha": stats["alpha"],
                "losses/alpha_loss": stats["alpha_loss"],
            }
            wandb.log(step_data)

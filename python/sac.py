# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import gym
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from diffusers.models import AutoencoderKL
from torch import optim
from torchvision import transforms
from tqdm import tqdm

import my_env
import wandb
from network import Actor, SoftQNetwork
from train_diffusion.mock_env import MockMineRL  # noqa: F401


@dataclass
class ReplayBufferData:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        obs_shape: np.ndarray,
        action_shape: np.ndarray,
        device: torch.device,
    ) -> None:
        self.size = size
        self.action_shape = action_shape
        self.device = device

        self.observations = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.observations[self.idx] = obs
        self.next_observations[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int) -> ReplayBufferData:
        idx = np.random.randint(0, self.size if self.full else self.idx, size=batch_size)
        return ReplayBufferData(
            self.observations[idx],
            self.next_observations[idx],
            torch.Tensor(self.actions[idx]).to(self.device),
            torch.Tensor(self.rewards[idx]).to(self.device),
            torch.Tensor(self.dones[idx]).to(self.device),
        )


@dataclass
class Args:
    exp_name: str = ""
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    gpu_id: int = 0
    """the gpu id to use"""

    # Algorithm specific arguments
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    gamma: float = 0.9
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    learning_starts: int = 500
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""




if __name__ == "__main__":
    args = tyro.cli(Args)
    args.total_timesteps = my_env.TIMEOUT
    run_name = f"SAC_{args.exp_name}"

    wandb.init(
        project="MineRL",
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.cuda.set_device(args.gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    datetime_str = time.strftime("%Y%m%d_%H%M%S")
    save_dir = Path("../train_result") / datetime_str
    save_dir.mkdir(parents=True, exist_ok=True)
    save_obs_dir = save_dir / "obs"
    save_obs_dir.mkdir(parents=True, exist_ok=True)
    save_action_dir = save_dir / "action"
    save_action_dir.mkdir(parents=True, exist_ok=True)
    save_reward_dir = save_dir / "reward"
    save_reward_dir.mkdir(parents=True, exist_ok=True)

    # env setup
    env = gym.make("MineRLMySetting-v0")
    # env = MockMineRL()  # noqa: ERA001
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    action_dim = 24
    image_h = 360 // 5
    image_w = 640 // 5
    z_h = image_h // 8
    z_w = image_w // 8
    z_ch = 4
    input_dim = z_h * z_w * z_ch

    actor = Actor(input_dim=input_dim, hidden_dim=256, action_dim=24, use_normalize=False)
    actor = actor.to(device)
    qf1 = SoftQNetwork(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=256,
        use_normalize=False,
    ).to(device)
    qf2 = SoftQNetwork(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=256,
        use_normalize=False,
    ).to(device)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    target_entropy = -action_dim
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)

    torch.autograd.set_detect_anomaly(True)

    rb = ReplayBuffer(
        args.buffer_size,
        np.array([image_h, image_w, 3]),
        np.array([24]),
        device,
    )
    start_time = time.time()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ],
    )

    # start the game
    obs = env.reset()
    obs = obs["pov"]
    obs = cv2.resize(obs, (image_w, image_h))
    reward = 0
    prev_inventory_action = 0
    progress_bar = tqdm(range(args.total_timesteps), dynamic_ncols=True)
    for global_step in range(args.total_timesteps):
        # put action logic here
        if global_step < args.learning_starts:
            env_action = env.action_space.sample()
            base_action = my_env.dict_action_to_array_action(env_action)
        else:
            obs_tensor = transform(obs)
            obs_tensor = obs_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                latent_image = vae.encode(obs_tensor).latent_dist.sample().mul_(0.18215)
                latent = latent_image.flatten(start_dim=1)
            base_action, _, _ = actor.get_action(latent)
            base_action = base_action[0].detach().cpu().numpy()
            env_action = my_env.array_action_to_dict_action(base_action)

        if prev_inventory_action == 1:
            env_action["inventory"] = 0
        prev_inventory_action = env_action["inventory"]

        # save
        cv2.imwrite(
            str(save_obs_dir / f"{global_step:08d}.png"),
            cv2.cvtColor(obs, cv2.COLOR_RGB2BGR),
        )
        with (save_action_dir / f"{global_step:08d}.json").open("w") as f:
            action_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in env_action.items()
            }
            json.dump(action_serializable, f)
        with (save_reward_dir / f"{global_step:08d}.json").open("w") as f:
            f.write(str(reward))

        # execute the game and log data.
        next_obs, reward, termination, info = env.step(env_action)
        env.render()
        next_obs = next_obs["pov"]
        next_obs = cv2.resize(next_obs, (image_w, image_h))
        rb.add(obs, next_obs, base_action, reward, termination)
        wandb.log({"reward": reward})

        if termination:
            break
        else:
            obs = next_obs

        # training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            data.observations = (
                torch.tensor(data.observations, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            )
            data.observations = (data.observations - 0.5) / 0.5
            data.observations = data.observations.to(device)
            data.next_observations = (
                torch.tensor(data.next_observations, dtype=torch.float32).permute(0, 3, 1, 2)
                / 255.0
            )
            data.next_observations = (data.next_observations - 0.5) / 0.5
            data.next_observations = data.next_observations.to(device)

            with torch.no_grad():
                data.observations = vae.encode(data.observations).latent_dist.sample().mul_(0.18215)
                data.observations = data.observations.flatten(start_dim=1)
                data.next_observations = (
                    vae.encode(data.next_observations).latent_dist.sample().mul_(0.18215)
                )
                data.next_observations = data.next_observations.flatten(start_dim=1)

                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1(data.next_observations, next_state_actions)
                qf2_next_target = qf2(data.next_observations, next_state_actions)
                min_q = torch.min(qf1_next_target, qf2_next_target)
                min_qf_next_target = min_q - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target
                ).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            pi, log_pi, _ = actor.get_action(data.observations)
            qf1_pi = qf1(data.observations, pi)
            qf2_pi = qf2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            with torch.no_grad():
                _, log_pi, _ = actor.get_action(data.observations)
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            a_optimizer.zero_grad()
            alpha_loss.backward()
            a_optimizer.step()
            alpha = log_alpha.exp().item()

            if global_step % 100 == 0:
                elapsed_time = time.time() - start_time
                data_dict = {
                    "charts/global_step": global_step,
                    "charts/elapse_time_sec": elapsed_time,
                    "losses/min_q": min_q.mean().item(),
                    "losses/min_qf_next_target": min_qf_next_target.mean().item(),
                    "losses/next_q_value": next_q_value.mean().item(),
                    "losses/qf1_values": qf1_a_values.mean().item(),
                    "losses/qf2_values": qf2_a_values.mean().item(),
                    "losses/qf1_loss": qf1_loss.item(),
                    "losses/qf2_loss": qf2_loss.item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item(),
                    "losses/alpha": alpha,
                    "losses/log_pi": log_pi.mean().item(),
                    "losses/alpha_loss": alpha_loss.item(),
                }
                wandb.log(data_dict)

        progress_bar.update(1)

    env.close()

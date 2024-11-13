"""MineRLで生成したデータについてのDataset."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

if TYPE_CHECKING:
    from pathlib import Path


def action_dict_to_tensor(action_dict: dict) -> torch.Tensor:
    # cameraを先頭に入れて、それ以外はonehotベクトルにする
    action = [
        action_dict["camera"][0] / 180,
        action_dict["camera"][1] / 180,
        action_dict["attack"],
        action_dict["back"],
        action_dict["drop"],
        action_dict["forward"],
        action_dict["hotbar.1"],
        action_dict["hotbar.2"],
        action_dict["hotbar.3"],
        action_dict["hotbar.4"],
        action_dict["hotbar.5"],
        action_dict["hotbar.6"],
        action_dict["hotbar.7"],
        action_dict["hotbar.8"],
        action_dict["hotbar.9"],
        action_dict["inventory"],
        action_dict["jump"],
        action_dict["left"],
        action_dict["pickItem"],
        action_dict["right"],
        action_dict["sneak"],
        action_dict["sprint"],
        action_dict["swapHands"],
        action_dict["use"],
    ]
    return torch.tensor(action, dtype=torch.float32)


def action_tensor_to_dict(action_tensor: torch.Tensor) -> dict:
    action_tensor = action_tensor.squeeze()
    return {
        "camera": [action_tensor[0].item() * 180, action_tensor[1].item() * 180],
        "attack": action_tensor[2].item(),
        "back": action_tensor[3].item(),
        "drop": action_tensor[4].item(),
        "forward": action_tensor[5].item(),
        "hotbar.1": action_tensor[6].item(),
        "hotbar.2": action_tensor[7].item(),
        "hotbar.3": action_tensor[8].item(),
        "hotbar.4": action_tensor[9].item(),
        "hotbar.5": action_tensor[10].item(),
        "hotbar.6": action_tensor[11].item(),
        "hotbar.7": action_tensor[12].item(),
        "hotbar.8": action_tensor[13].item(),
        "hotbar.9": action_tensor[14].item(),
        "inventory": action_tensor[15].item(),
        "jump": action_tensor[16].item(),
        "left": action_tensor[17].item(),
        "pickItem": action_tensor[18].item(),
        "right": action_tensor[19].item(),
        "sneak": action_tensor[20].item(),
        "sprint": action_tensor[21].item(),
        "swapHands": action_tensor[22].item(),
        "use": action_tensor[23].item(),
    }


class MineRLDataset(VisionDataset):
    """MineRLで生成したデータについてのDataset."""

    def __init__(
        self,
        root: Path,
        image_size: int,
        seq_len: int,
    ) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ],
        )
        super().__init__(
            root,
            transform=transform,
            target_transform=None,
        )
        self.trial_list = sorted(root.glob("*"))
        self.data_list = []
        for trial in self.trial_list:
            obs_list = sorted((trial / "obs").glob("*.png"))
            action_list = sorted((trial / "action").glob("*.json"))
            obs_list = obs_list[: len(action_list)]

            self.data_list += zip(obs_list, action_list)
        self.seq_len = seq_len

    def __getitem__(self, index: int) -> list[tuple[Any, Any]]:
        """Get item.

        Args:
            index (int): Index

        Returns:
            list of tuple: [(image_t, action_t),
                            (image_t+1, action_t+1),
                            ...,
                            (image_t+seq_len-1, action_t+seq_len-1)]

        """
        image_list = []
        action_list = []
        for i in range(self.seq_len):
            obs_path, action_path = self.data_list[index + i]
            obs = Image.open(obs_path)
            action_dict = json.load(action_path.open())

            """
            Example of action:
               {'ESC': 0, 'attack': 1, 'back': 1, 'camera': [-68.02954864501953, 89.1246566772461],
               'drop': 1, 'forward': 0, 'hotbar.1': 1, 'hotbar.2': 1, 'hotbar.3': 0, 'hotbar.4': 1,
               'hotbar.5': 1, 'hotbar.6': 0, 'hotbar.7': 1, 'hotbar.8': 0, 'hotbar.9': 1,
               'inventory': 1, 'jump': 0, 'left': 0, 'pickItem': 1, 'right': 0, 'sneak': 0,
               'sprint': 0, 'swapHands': 0, 'use': 0}
            """
            action = action_dict_to_tensor(action_dict)

            if self.transform is not None:
                obs = self.transform(obs)
            if self.target_transform is not None:
                action = self.target_transform(action)
            image_list.append(obs)
            action_list.append(action)
        image_tensor = torch.stack(image_list)
        action_tensor = torch.stack(action_list)

        return image_tensor, action_tensor

    def __len__(self) -> int:
        return len(self.data_list) - self.seq_len + 1

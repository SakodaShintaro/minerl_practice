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


class MineRLDataset(VisionDataset):
    """MineRLで生成したデータについてのDataset."""

    def __init__(
        self,
        root: Path,
        image_size: int,
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
        self.seq_len = 3

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
            action = json.load(action_path.open())

            """
            Example of action:
               {'ESC': 0, 'attack': 1, 'back': 1, 'camera': [-68.02954864501953, 89.1246566772461],
               'drop': 1, 'forward': 0, 'hotbar.1': 1, 'hotbar.2': 1, 'hotbar.3': 0, 'hotbar.4': 1,
               'hotbar.5': 1, 'hotbar.6': 0, 'hotbar.7': 1, 'hotbar.8': 0, 'hotbar.9': 1,
               'inventory': 1, 'jump': 0, 'left': 0, 'pickItem': 1, 'right': 0, 'sneak': 0,
               'sprint': 0, 'swapHands': 0, 'use': 0}
            """
            # cameraを先頭に入れて、それ以外はonehotベクトルにする
            # action = [
            #     action["camera"][0] / 180,
            #     action["camera"][1] / 180,
            #     action["attack"],
            #     action["back"],
            #     action["drop"],
            #     action["forward"],
            #     action["hotbar.1"],
            #     action["hotbar.2"],
            #     action["hotbar.3"],
            #     action["hotbar.4"],
            #     action["hotbar.5"],
            #     action["hotbar.6"],
            #     action["hotbar.7"],
            #     action["hotbar.8"],
            #     action["hotbar.9"],
            #     action["inventory"],
            #     action["jump"],
            #     action["left"],
            #     action["pickItem"],
            #     action["right"],
            #     action["sneak"],
            #     action["sprint"],
            #     action["swapHands"],
            #     action["use"],
            # ]
            action = [action["inventory"] for i in range(24)]
            action = torch.tensor(action, dtype=torch.float32)

            if self.transform is not None:
                obs = self.transform(obs)
            if self.target_transform is not None:
                action = self.target_transform(action)
            image_list.append(obs)
            action_list.append(action)
        image_tensor = torch.stack(image_list)
        action_tensor = torch.stack(action_list)

        # actionを修正する
        for i in range(self.seq_len - 1, 0, -1):
            curr = action_tensor[i].mean()
            prev = action_tensor[i - 1].mean()
            if curr == 1.0 and prev == 1.0:
                action_tensor[i].fill_(0.0)

        image_tensor = image_tensor[1:]
        action_tensor = action_tensor[1:]

        return image_tensor, action_tensor

    def __len__(self) -> int:
        return len(self.data_list) - self.seq_len + 1

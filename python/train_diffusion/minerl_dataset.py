"""MineRLで生成したデータについてのDataset."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from PIL import Image
from torchvision.datasets import VisionDataset

if TYPE_CHECKING:
    from pathlib import Path


class MineRLDataset(VisionDataset):
    """MineRLで生成したデータについてのDataset."""

    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
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
        result = []
        for i in range(self.seq_len):
            obs_path, action_path = self.data_list[index + i]
            obs = Image.open(obs_path)
            action = json.load(action_path.open())
            if self.transform is not None:
                obs = self.transform(obs)
            if self.target_transform is not None:
                action = self.target_transform(action)
            result.append((obs, action))
        return result

    def __len__(self) -> int:
        return len(self.data_list) - self.seq_len + 1

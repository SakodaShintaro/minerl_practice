"""Test the MineRLDataset class."""

import argparse
from pathlib import Path

from minerl_dataset import MineRLDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--image_size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir
    image_size = args.image_size

    dataset = MineRLDataset(root_dir, image_size)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    print(len(dataset))

    for batch in loader:
        print(type(batch), len(batch))
        image, action = batch
        print(image.shape, image.dtype, image.min(), image.max())
        print(action.shape, action.dtype, action.min(), action.max())

        action = action[:, 0]
        image = image[:, 0]

        # 各imageにactionをテキストで左上に書き込む

        # 画像を保存
        save_image(
            image,
            "image.png",
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
        )

        break

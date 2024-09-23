"""Test the MineRLDataset class."""

import argparse
from pathlib import Path

from minerl_dataset import MineRLDataset
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ],
    )
    dataset = MineRLDataset(root_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    print(len(dataset))
    print(dataset[0])

    for batch in loader:
        print(type(batch), len(batch))
        image, action = batch
        print(image.shape, image.dtype, image.min(), image.max())
        print(action.shape, action.dtype, action.min(), action.max())
        break

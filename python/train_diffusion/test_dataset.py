"""Test the MineRLDataset class."""

import argparse
from pathlib import Path

import torch
from minerl_dataset import MineRLDataset
from PIL import ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def add_text_to_image(image_tensor: torch.Tensor, action_tensor: torch.Tensor) -> ImageDraw:
    """画像にアクションをテキストで書き込む."""
    # torch.TensorからPIL.Imageに変換
    image_pil = transforms.ToPILImage()(image_tensor.cpu())

    # 描画用のDrawオブジェクト作成
    draw = ImageDraw.Draw(image_pil)

    # Actionをテキストとして取得
    action_text = f"Action: {action_tensor[0].item()}"

    # フォントが指定されていない場合はデフォルトフォントを使用
    font = ImageFont.load_default(size=32)

    # テキストを左上に描画
    draw.text((5, 5), action_text, font=font, fill=(255, 255, 255))

    return image_pil


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir
    image_size = args.image_size

    dataset = MineRLDataset(root_dir, image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    print(len(dataset))

    save_dir = Path("images")
    save_dir.mkdir(exist_ok=True)

    for batch in loader:
        print(type(batch), len(batch))
        images, actions = batch
        print(images.shape, images.dtype, images.min(), images.max())
        print(actions.shape, actions.dtype, actions.min(), actions.max())

        for i in range(images.size(0)):
            image = images[i, 0]
            action = actions[i, 0]
            image_with_text = add_text_to_image(image, action)
            image_with_text.save(save_dir / f"image_{i:08d}.png")

        break

"""出力画像を連結するスクリプト."""

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_dir = args.result_dir

    dir_gt = result_dir / "gt"
    dir_pred = result_dir / "predict"

    save_dir = result_dir / "concat"
    save_dir.mkdir(exist_ok=True)

    gt_paths = sorted(dir_gt.glob("*.png"))
    pred_paths = sorted(dir_pred.glob("*.png"))

    curr_images = []
    prev_id = 0
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        curr_id = int(gt_path.stem)
        gt = cv2.imread(str(gt_path))
        pred = cv2.imread(str(pred_path))
        # 画像を横に連結
        concat = cv2.hconcat([gt, pred])

        if curr_id == prev_id + 1:
            # idが連続している場合は追加
            curr_images.append(concat)
        else:
            # これまでのものを縦に連結して保存
            print(f"{prev_id=}, {curr_id=}, {len(curr_images)=}")
            save_image = cv2.vconcat(curr_images)
            save_path = save_dir / gt_path.name
            cv2.imwrite(str(save_path), save_image)
            curr_images = [concat]
        prev_id = curr_id

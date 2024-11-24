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

    save_dir = result_dir / "eval_concat"
    save_dir.mkdir(exist_ok=True)

    dir_gt = result_dir / "eval_gt"

    nfe_list = [
        64, 16, 4, 1
    ]
    dir_pred_list = [result_dir / f"eval_predict_{nfe}" for nfe in nfe_list]

    gt_paths = sorted(dir_gt.glob("*.png"))
    pred_paths_list = [sorted(dir_pred.glob("*.png")) for dir_pred in dir_pred_list]

    all_image_list = []

    for i in range(len(gt_paths)):
        gt_path = gt_paths[i]
        pred_paths = [pred_paths[i] for pred_paths in pred_paths_list]

        image_list = [cv2.imread(str(gt_path))]
        for pred_path in pred_paths:
            image_list.append(cv2.imread(str(pred_path)))

        image = cv2.hconcat(image_list)
        save_path = save_dir / gt_path.name
        cv2.imwrite(str(save_path), image)
        all_image_list.append(image)

    all_image = cv2.vconcat(all_image_list)
    save_path = save_dir / "all.png"
    cv2.imwrite(str(save_path), all_image)

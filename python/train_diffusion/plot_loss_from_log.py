"""log.txtから損失の推移をプロットするスクリプト."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_path = args.log_path

    step_list = []
    loss_list = []
    with log_path.open() as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if "Train Loss" not in line:
                continue
            # [2024-09-23 12:13:55] (step=0047800) Train Loss: 0.1560, Train Steps/Sec: 1.86
            elements = line.split(" ")
            step = int(elements[2].replace("(step=", "").replace(")", ""))
            loss = float(elements[5].replace(",", ""))
            step_list.append(step)
            loss_list.append(loss)


    plt.plot(step_list, loss_list)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    save_path = log_path.parent / "step_loss.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Save plot to {save_path}")

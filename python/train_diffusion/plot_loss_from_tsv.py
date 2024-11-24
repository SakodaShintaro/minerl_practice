"""log.tsvから損失の推移をプロットするスクリプト."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_path = args.log_path
    df = pd.read_csv(log_path, sep="\t")

    print(df.head())

    # デフォルトのcmapを使用
    cmap = plt.get_cmap("tab10")

    plt.subplot(2, 1, 1)
    plt.plot(df["step"], df["loss_fm"], label="flow matching loss", color=cmap(0))
    plt.ylabel("Flow Matching Loss")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(df["step"], df["loss_sc"], label="shortcut loss", color=cmap(1))
    plt.ylabel("Shortcut Loss")
    plt.grid()

    plt.xlabel("Step")
    plt.tight_layout()

    save_path = log_path.parent / "step_loss.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Save plot to {save_path}")

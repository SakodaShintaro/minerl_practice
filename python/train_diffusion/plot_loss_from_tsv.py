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

    plt.plot(df["step"], df["loss_fm"], label="Flow Matching Loss")
    plt.plot(df["step"], df["loss_sc"], label="Shortcut Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid()
    save_path = log_path.parent / "step_loss.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Save plot to {save_path}")

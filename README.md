# minerl_practice

MineRLの環境でいろいろ試してみるリポジトリ

## 環境構築の仕方

`docker` ディレクトリのスクリプトを参照

## その他メモ

まずは既存の強化学習ライブラリが動くのかどうかを試す。

[stable-baselines3](https://github.com/DLR-RM/stable-baselines3) の方を試した。

```bash
pip3 install stable-baselines3[extra]
```

行動・観測空間を適切に変換することで動作はした。詳細は `python/train_sb3.py` 参照。

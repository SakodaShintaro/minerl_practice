# minerl_practice

MineRLの環境でいろいろ試してみるリポジトリ

## 環境構築の仕方

(1) 以下リポジトリの `docker` ディレクトリのスクリプトを参照してコンテナを作る

- <https://github.com/SakodaShintaro/misc/tree/main/docker>

(2) その後 `./setup.sh` を実行する（15分くらいかかる）

(3) `pip3 install -r ./python/requirements.txt`

## メモ

意外と拡散モデルでの次状態予測を書く上で考えるべきことが多い。入力のタイムステップをどれくらいにするか、予測のタイムステップをどれくらいにするか。
ニューラルネットワークのモデルもタイムステップが入るので画像部分でいろいろいじる必要がある。
また、行動の埋め込みも同様にタイムステップがあるのでこれをどうすると綺麗なのかは不明。

元画像サイズ: 360 * 640

actionのdimは24

### MineRLを編集する

`$HOME/.local/lib/python3.10/site-packages/minerl/herobraine/env_specs/obtain_specs.py`

を変更してタイムアウトステップなどを変えることができる。

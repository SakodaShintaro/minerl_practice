# NVIDIAの公式イメージをベースにする
FROM nvcr.io/nvidia/pytorch:24.06-py3

# 環境変数の設定
ENV USER_NAME=user
ENV USER_UID=1000
ENV USER_GID=1000

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y sudo

# ユーザーとグループの作成
RUN groupadd --gid $USER_GID $USER_NAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME && \
    echo "$USER_NAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME

# ワーキングディレクトリの設定
WORKDIR /home/$USER_NAME

# ユーザーの切り替え
USER $USER_NAME

# デフォルトのコマンド
CMD ["/bin/bash"]

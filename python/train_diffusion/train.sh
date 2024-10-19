#!/bin/bash
set -eux

script_dir=$(readlink -f $(dirname $0))
cd ${script_dir}

result_dir=${script_dir}/../../train_result/result_$(date +"%Y%m%d_%H%M%S")

MODEL_NAME="DiT-S/2"
DATA_PATH="../../data"

python3 train.py \
  --model=${MODEL_NAME} \
  --data_path=${DATA_PATH} \
  --results_dir=${result_dir} \
  --use_flow_matching

# checkpointディレクトリの中から最新のものを指定
ckpt_id=$(ls ${result_dir}/checkpoints | sort | tail -n 1)

python3 sample.py \
  --model=${MODEL_NAME} \
  --data_path=${DATA_PATH} \
  --ckpt=${result_dir}/checkpoints/${ckpt_id}

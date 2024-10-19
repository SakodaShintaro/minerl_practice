#!/bin/bash
set -eux

script_dir=$(readlink -f $(dirname $0))
cd ${script_dir}

result_dir=${script_dir}/../../train_result/result_$(date +"%Y%m%d_%H%M%S")

MODEL_NAME="DiT-S/2"
DATA_PATH="../../data"

python3 train_flow_matching.py \
  --model=${MODEL_NAME} \
  --data_path=${DATA_PATH} \
  --results_dir=${result_dir}

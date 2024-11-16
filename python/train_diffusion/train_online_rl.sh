#!/bin/bash
set -eux

script_dir=$(readlink -f $(dirname $0))
cd ${script_dir}

result_dir=${script_dir}/../../train_result/$(date +"%Y%m%d_%H%M%S")_online_rl

MODEL_NAME="DiT-S/2"

python3 main.py \
  --model=${MODEL_NAME} \
  --results_dir=${result_dir}

python3 plot_images.py ${result_dir}

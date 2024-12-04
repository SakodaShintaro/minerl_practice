#!/bin/bash
set -eux

script_dir=$(readlink -f $(dirname $0))
cd ${script_dir}

result_dir=${script_dir}/../../train_result/$(date +"%Y%m%d_%H%M%S")_batch_sl

MODEL_NAME="DiT-S/2"
DATA_PATH="../../dataset/skip_frame/"

python3 train_batch_sl.py \
  --model=${MODEL_NAME} \
  --data_path=${DATA_PATH} \
  --results_dir=${result_dir}

python3 plot_images.py ${result_dir}
python3 plot_loss_from_tsv.py ${result_dir}/log.tsv

exit 0

python3 evaluate_trained_model.py ${result_dir}/checkpoints/00100000.pt --nfe 64
python3 evaluate_trained_model.py ${result_dir}/checkpoints/00100000.pt --nfe 16
python3 evaluate_trained_model.py ${result_dir}/checkpoints/00100000.pt --nfe 4
python3 evaluate_trained_model.py ${result_dir}/checkpoints/00100000.pt --nfe 1
python3 plot_images_each_nfe.py ${result_dir}

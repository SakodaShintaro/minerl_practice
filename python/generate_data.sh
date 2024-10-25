#!/bin/bash
set -eux

script_dir=$(readlink -f $(dirname $0))
result_dir=$(readlink -f $1)
cd ${script_dir}

for i in $(seq 0 99); do
    # 0埋め4桁
    dir_name=$(printf "%04d" $i)
    save_dir=${result_dir}/$dir_name
    mkdir -p $save_dir
    python3 main.py $save_dir
done

#!/bin/bash
set -eux

docker run --gpus all \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v $HOME/data:/home/user/data/ \
            -v $HOME/work:/home/user/work/ \
            -v $HOME/.cache/:/home/user/.cache/ \
            -v /media:/media \
            -p 7007:7007 \
            -it \
            --ipc=host \
            --privileged \
            $1 \
            /bin/bash

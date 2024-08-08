#!/bin/bash

xhost +
docker run --rm -e DISPLAY=$DISPLAY -v /home/intel-admin/.Xauthority:/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -it -v `pwd`:/savedir --privileged --net host --ipc=host yolo-demo-openvino:1.0 /bin/bash

# python track.py | stdbuf -oL sed -n -e 's/^.*, //p' | stdbuf -oL cut -d , -f 1 | grep -e '[0-9]*ms' > perf.log

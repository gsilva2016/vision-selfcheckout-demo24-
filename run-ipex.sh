#!/bin/bash

xhost +
docker run --rm -e DISPLAY=$DISPLAY -v /home/intel-admin/.Xauthority:/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -it -v `pwd`:/savedir --privileged --net host --ipc=host yolo-demo-torch-ipex:1.0 /bin/bash


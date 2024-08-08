
docker run --rm -e DISPLAY=$DISPLAY --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -it -v `pwd`:/savedir --gpus all --runtime=nvidia --net host --ipc=host yolo_ov_demo /bin/bash

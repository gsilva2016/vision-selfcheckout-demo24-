#!/bin/bash

if [ "$INCLUDE_REALSENSE" != "Y" ]
then
	INCLUDE_REALSENSE=N
fi

if [ "$INCLUDE_NPU" != "Y" ]
then
	INCLUDE_NPU=N
fi

docker build -t yolo-demo-openvino:1.0 --build-arg "INCLUDE_NPU=$INCLUDE_NPU" --build-arg "INCLUDE_REALSENSE=$INCLUDE_REALSENSE" -f Dockerfile.openvino .

docker build -t yolo-demo-torchopenvino:1.0 --build-arg "INCLUDE_NPU=$INCLUDE_NPU" --build-arg "INCLUDE_REALSENSE=$INCLUDE_REALSENSE" -f Dockerfile.torchopenvino .

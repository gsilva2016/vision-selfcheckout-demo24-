#!/bin/bash

if [ "$INCLUDE_REALSENSE" != "Y" ]
then
	INCLUDE_REALSENSE=N
fi

if [ "$INCLUDE_NPU" != "Y" ]
then
	INCLUDE_NPU=N
fi

if [ "$INCLUDE_YOLONAS" != "Y" ]
then
	INCLUDE_YOLONAS=N
fi

docker build -t yolo-demo-openvino:1.0 --build-arg "INCLUDE_YOLONAS=$INCLUDE_YOLONAS" --build-arg "INCLUDE_NPU=$INCLUDE_NPU" --build-arg "INCLUDE_REALSENSE=$INCLUDE_REALSENSE" -f Dockerfile.openvino .

docker build -t yolo-demo-torch:1.0 --build-arg "INCLUDE_YOLONAS=$INCLUDE_YOLONAS" --build-arg "USE_IPEX=N" --build-arg "INCLUDE_NPU=$INCLUDE_NPU" --build-arg "INCLUDE_REALSENSE=$INCLUDE_REALSENSE" -f Dockerfile.torch .

docker build -t yolo-demo-torch-ipex:1.0 --build-arg "INCLUDE_YOLONAS=$INCLUDE_YOLONAS" --build-arg "INCLUDE_NPU=$INCLUDE_NPU" --build-arg "INCLUDE_REALSENSE=$INCLUDE_REALSENSE" --build-arg "USE_IPEX=Y" -f Dockerfile.torch .

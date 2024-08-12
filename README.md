# Vision Self-Checkout CV Demos

Vision Self-Checkout CV Demos using yolov8, resnet-50, BiT, and efficientnet. This repo demonstrates various Media+AI system architectures for inference and are easily launched from containers. 

## Python OpenVINO Demos

### OpenVINO Inference of GPU accelerated media via OpenCV 

Build Container with CPU, GPU, NPU support: 

```
INCLUDE_NPU=Y ./build.sh
```

Build Container with CPU, GPU support: 

```
./build.sh
```

Run Container:

```
./run-openvino.sh
```

Run GUI Application:

```
python3 openvino-demo.py --source <MY_VIDEO>.mp4 --enable_int8 --device "GPU" --cls_model efficientnet-b0 --show
```

Run Console Application:

```
python3 openvino-demo.py --source <MY_VIDEO>.mp4 --enable_int8 --device "GPU" --cls_model resnet-50
```

### OpenVINO Inference of RealSense Camera media

Build Container with CPU, GPU, NPU and RealSense support: 

```
INCLUDE_NPU=Y INCLUDE_REALSENSE=Y ./build.sh
```

Run Container:

```
./run-openvino.sh
```

Run GUI Application:

```
rs list
```

```
serial_id=130322273236
python3 openvino-demo.py --source $serial_id --enable_int8 --device "GPU" --cls_model resnet-50 --print_metrics_interval 15 --show
```

## Python PyTorch OpenVINO backend Demos

TODO - https://github.com/gsilva2016/llmrts-intel/tree/ipex-openvino-yolo-xpu

## Python OpenVINO + Weaviate Inference Microservice Demo

This demonstrates using OpenVINO for Yolov8 object detection and after performing resnet-50 classification via a PyTorch Weaviate microservice. The PyTorch microservice performs PyTorch XPU inference but also capable of CUDA gpu inference as well.

TODO - https://github.com/weaviate/i2v-pytorch-models/pull/10

## Python OpenVINO + Weaviate Feature Search Microservice Demo

This demonstrates using OpenVINO for Yolov8 object detection and after performing microservice call for a feature vector search via Weaviate's custom engine and APIs. The features are generated from resnet-50 using either PyTorch XPU accelerated inference or OpenVINO CPU, GPU, NPU accelerated inference. The Weaviate microservice is also capable of CUDA inference as well.

TODO - 

https://github.com/gsilva2016/weaviate-examples/tree/self-checkout-demo/nearest-neighbor-bottle-search
https://github.com/gsilva2016/i2v-pytorch-models/tree/intel_igpu_support

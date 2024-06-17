# only CPU works with this image
#FROM ultralytics/ultralytics:8.1.17-cpu
#FROM ultralytics/ultralytics:8.1.29-cpu
#FROM openvino/ubuntu22_dev:2024.0.0
FROM ubuntu:22.04

WORKDIR /yolo_ov_demo
user root
ENV DEBIAN_FRONTEND=noninteractive
#ARG BUILD_DEPENDENCIES="vim libgtk2.0-dev pkg-config python3-opencv python3-tk"
ARG BUILD_DEPENDENCIES="vim libgtk2.0-dev pkg-config python3-dev python3-tk gpg wget python3-pip"
RUN apt -y update && \
    apt install -y ${BUILD_DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
RUN apt -y update && apt install -y wget && \
    rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

# Intel gfx/media client drivers
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg; echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | tee /etc/apt/sources.list.d/intel-gpu-jammy.list; apt update
ARG BUILD_DEPENDENCIES="nasm yasm libmfx-dev libva-dev vim libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio gstreamer1.0-vaapi libmfx1 libmfxgen1 libmfx-tools libvpl2 libva-drm2 libva-x11-2 libva-wayland2 libva-glx2 vainfo intel-media-va-driver-non-free ffmpeg build-essential git pkg-config python3-dev cmake pkg-config python3-opencv unzip"
RUN apt -y update && \
    apt install -y ${BUILD_DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

# OpenCV build with HWA and DNN modules
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null; echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list; apt update -y; apt install -y intel-basekit && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

RUN wget https://github.com/opencv/opencv/archive/refs/tags/4.9.0.zip && unzip 4.9.0.zip; cd opencv-4.9.0/; mkdir -p build; cd build; cmake -DBUILD_opencv_dnn=ON  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_TESTS=OFF -DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.10/dist-packages/numpy/core/include -DOPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.10/dist-packages -DPYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages -DOPENCV_PYTHON3_INSTALL_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") -DPYTHON_EXECUTABLE=$(which python3) -DBUILD_DOCS=OFF -DVIDEOIO_PLUGIN_LIST="mfx;ffmpeg" ..; cmake --build . --config Release -- -j`nproc`; make install; echo $(python3 -c "import cv2; print(cv2.__version__)")

#RUN wget -O sample.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4 
#RUN wget -O sample.mp4 https://download.pexels.com/vimeo/421906504/pexels-karolina-grabowska-4465029.mp4?width=1920
RUN pip install pyrealsense2
RUN pip install openvino==2024.1.0
RUN pip install "paho-mqtt<2.0.0" ultralytics==8.1.29
RUN pip install "mpmath<1.4.0"
RUN pip install weaviate-client
RUN pip install tensorflow
COPY vision24-demo.py .
COPY allow_x11_display_docker_openvino.patch /usr/local/lib/python3.10/dist-packages/ultralytics/utils
RUN cd /usr/local/lib/python3.10/dist-packages/ultralytics/utils; patch < allow_x11_display_docker_openvino.patch
COPY allow_openvino_device_change.patch /usr/local/lib/python3.10/dist-packages/ultralytics/nn
RUN cd /usr/local/lib/python3.10/dist-packages/ultralytics/nn; patch < allow_openvino_device_change.patch
COPY allow_openvino_npu_quant_change.patch2 /usr/local/lib/python3.10/dist-packages/ultralytics/nn
RUN cd /usr/local/lib/python3.10/dist-packages/ultralytics/nn; patch < allow_openvino_npu_quant_change.patch2

RUN mkdir npu-drivers; cd npu-drivers; wget https://github.com/intel/linux-npu-driver/releases/download/v1.1.0/intel-driver-compiler-npu_1.1.0.20231117-6904283384_ubuntu22.04_amd64.deb; wget https://github.com/intel/linux-npu-driver/releases/download/v1.1.0/intel-fw-npu_1.1.0.20231117-6904283384_ubuntu22.04_amd64.deb; wget https://github.com/intel/linux-npu-driver/releases/download/v1.1.0/intel-level-zero-npu_1.1.0.20231117-6904283384_ubuntu22.04_amd64.deb; wget https://github.com/oneapi-src/level-zero/releases/download/v1.10.0/level-zero_1.10.0+u22.04_amd64.deb; dpkg -i *.deb
#RUN cd /opt/intel/openvino/samples/cpp; ./build_samples.sh

COPY run-usb-camera.sh .
COPY run-rtsp-camera.sh .
COPY multiproc.py .
COPY run-2rtsp-camera.sh .

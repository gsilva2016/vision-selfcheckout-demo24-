#!/bin/bash
pip install pyrealsense2 realsense-cli
cp -R /savedir/PRETRAINED_MODELS/BiT_M_R50x1_10C_50e_IR/ .
cp /savedir/vision24-demo.py .
python3 vision24-demo.py 

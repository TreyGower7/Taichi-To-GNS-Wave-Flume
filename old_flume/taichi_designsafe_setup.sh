#!/bin/bash

which python3
python3 -m pip install --upgrade pip
python3 -m pip install taichi==1.7.1
python3 -m pip install imageio
python3 -m pip install ffmpeg
python3 -m pip install -r requirements.txt

python3 -m pip list



# Assuming that mpm99_designsafe.py is in the same directory as this script
mkdir taichi-output
mkdir taichi-output/frames
export TAICHI_SCRIPT="mpm99_designsafe.py"
export INPUT_DIR="."
export OUTPUT_DIR="./taichi-output/"
python3 ${INPUT_DIR}/${TAICHI_SCRIPT}


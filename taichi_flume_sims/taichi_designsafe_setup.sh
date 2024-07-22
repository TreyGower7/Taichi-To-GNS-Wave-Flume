#!/bin/bash

which python3
python3 -m pip install --upgrade pip
python3 -m pip install taichi==1.7.1
python3 -m pip install imageio
python3 -m pip install ffmpeg
python3 -m pip install -r ../requirements.txt

# Check if numpy is installed and its version specific for the wave flume simulation
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)

if [ "$NUMPY_VERSION" != "1.23.5" ]; then
    python3 -m pip uninstall -y numpy
    python3 -m pip install numpy==1.23.5
fi

python3 -m pip list
# Assuming that mpm99_designsafe.py is in the same directory as this script
mkdir taichi-output
mkdir taichi-output/frames
export TAICHI_SCRIPT="mpm99_designsafe.py"
export INPUT_DIR="."
export OUTPUT_DIR="./taichi-output/"
python3 ${INPUT_DIR}/${TAICHI_SCRIPT}

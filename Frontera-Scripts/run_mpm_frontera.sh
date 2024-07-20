#!/bin/bash
module load python3/3.9.2
which python3

python3 -m pip install --upgrade pip
python3 -m pip install taichi==1.7.1
python3 -m pip install imageio
python3 -m pip install ffmpeg

# Check if numpy is installed and its version
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)

if [ "$NUMPY_VERSION" != "1.23.5" ]; then
    python3 -m pip uninstall -y numpy
    python3 -m pip install numpy==1.23.5
fi

python3 -m pip list

# Assuming that mpm99_designsafe.py is in the same directory as this script
mkdir taichi-output
mkdir taichi-output/frames

python3 mpm_frontera.py
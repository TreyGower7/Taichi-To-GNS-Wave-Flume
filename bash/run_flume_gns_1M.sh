#!/bin/bash

TMP_DIR="./"
VENV_PATH="${TMP_DIR}/venv"
CURRENT_DIR=$(pwd)
#echo $CURRENT_DIR

# Check if virtual environment exists
if [ "$CURRENT_DIR" != "/home/tg458981/work/HPCWork/gns" ]; then
  if [ ! -d "${VENV_PATH}" ]; then
    echo "Virtual environment does not exist; Running build_venv_frontera.sh"

    # Run the script to build the virtual environment
    if [ -f "./build_venv_frontera.sh" ]; then
      ./build_venv_frontera.sh
    else
      echo "build_venv_frontera.sh not found; Please ensure it is present in the current directory"
      exit 1
    fi
  fi
fi
# Activate the virtual environment
source "${VENV_PATH}/bin/activate"
if [ -d "gns-flume" ]; then
  echo "Directory gns-flume exists; Using that parent directory"
  TMP_DIR="./gns-flume"

  # Check if the Flume dataset directory exists
  if [ -d "./gns-flume/Flume" ]; then
    echo "Flume Dataset exists; Using that dataset"
    DATASET_NAME="Flume"
  else
    read -p "Enter the DATASET_NAME: " DATASET_NAME
  fi
else
  read -p "Enter the TMP_DIR: " TMP_DIR
  read -p "Enter the DATASET_NAME: " DATASET_NAME
fi

# Define paths
DATA_PATH="${TMP_DIR}/${DATASET_NAME}/dataset/"
MODEL_PATH="${TMP_DIR}/${DATASET_NAME}/models/"
ROLLOUT_PATH="${TMP_DIR}/${DATASET_NAME}/rollout/"

# Check if the models directory exists
if [ -d "${MODEL_PATH}" ]; then
  echo "Models directory exists; Using that directory for model storage"
else
  # Create the directory if it doesn't exist
  mkdir -p "${MODEL_PATH}"
fi

# Check if the rollout directory exists
if [ -d "${ROLLOUT_PATH}" ]; then
  echo "Rollout directory exists; Using that directory for results"
else
  # Create the directory if it doesn't exist
  mkdir -p "${ROLLOUT_PATH}"
fi

#read -p "Enter the number of training steps for the GNS model (e.g 10): " n_steps

# Run the python command
python3 -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=1000000 --n_gpus=4 --ntraining_steps=50000

python3 -m gns.train --mode="rollout" --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --output_path=${ROLLOUT_PATH} --model_file="model-100000.pt" --train_state_file="train_state-100000.pt"

python3 -m gns.render_rollout --output_mode="gif" --rollout_dir=${ROLLOUT_PATH} --rollout_name="rollout_ex0"

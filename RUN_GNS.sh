#!/bin/bash

# Allows for the easy use of GNS

# Check if the directory gns-flume exists

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

read -p "Enter the number of training steps for the GNS model (e.g 10): " n_steps

# Run the python command
python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --ntraining_steps=${n_steps}
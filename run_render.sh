
if [ -d "gns-flume" ]; then
  TMP_DIR="${TMP_DIR}"
  # Check if the Flume dataset directory exists
  if [ -d "./gns-flume/Flume" ]; then
    DATASET_NAME="${DATASET_NAME}"
  else
    echo "Ensure you have trained your model first and have defined paths"
    exit 1
  fi
fi

# Path to models folder 
cd ${TMP_DIR}/${DATASET_NAME}/models

model_name=$(ls | grep "model-$n_steps.pt")
train_state=$(ls | grep "train_state-$n_steps.pt")

# Generate Example
python3 -m gns.train --mode="rollout" --data_path="${DATA_PATH}" --model_path="${MODEL_PATH}" --output_path="${ROLLOUT_PATH}" --model_file="$model_name" --train_state_file="$train_state"

# Render gif
python3 -m gns.render_rollout --output_mode="gif" --rollout_dir="${ROLLOUT_PATH}" --rollout_name="rollout_ex0"

echo "Running Model: $model_name"

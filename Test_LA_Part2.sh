#!/bin/bash

# Directory containing .pth files
PTH_DIR="result/SAM3D_VNet_SSL/LA_16_SemiSupervised_V12/Pth_Part2"  # 修改为你存放模型的目录

# Directory to save logs
LOG_DIR="result/SAM3D_VNet_SSL/LA_16_SemiSupervised_V12/Test_Part2"  # 修改为你保存日志的目录

# Root path for dataset
ROOT_PATH="./datasets/LA"  # 修改为你数据集的路径

# Number of classes
NUM_CLASSES=2  # 修改为你的数据集类别数

# Number of outputs
NUM_OUTPUT=2  # Assuming this is the default number of outputs

# Test save path (if none, we just ignore it in this script)
TEST_SAVE_PATH="None"  # 如果不保存结果，可以保持None

# Ensure LOG_DIR exists, if not, create it
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

# Iterate over each .pth file in the directory
for MODEL_PATH in $PTH_DIR/*.pth; do
  # Extract epoch number from filename (optional)
  EPOCH=$(echo $MODEL_PATH | grep -oP '(?<=epoch_)\d+(?=_checkpoint.pth)')
  
  # Set log filename to include epoch number for logging
  LOG_FILE="$LOG_DIR/Test.log"

  echo "Testing $MODEL_PATH, logging to $LOG_FILE"

  # Call the Python script with the current .pth file
  python -m Test_LA_Part2 \
    --model_name "SAMV12" \
    --model_load "$MODEL_PATH" \
    --log_path "$LOG_FILE" \
    --test_save_path "$TEST_SAVE_PATH" \
    --root_path "$ROOT_PATH" \
    --num_classes "$NUM_CLASSES" \
    --num_outputs "$NUM_OUTPUT"  # Assuming this is the default number of outputs
done

#!/bin/bash

# 存放 .pth 文件的目录
# PTH_DIR="result/SAM3D_VNet_SSL/LA_16_SemiSupervised_V14_2_2/Pth_Part1"  # 修改为你存放模型的目录
PTH_DIR="./result/SAM3D_VNet_SSL/LA_8_SemiSupervised_V15_2/Pth_Part1"

# 保存日志的目录
# LOG_DIR="result/SAM3D_VNet_SSL/LA_16_SemiSupervised_V14_2_2/Test_Part1"  # 修改为你保存日志的目录
LOG_DIR="./result/SAM3D_VNet_SSL/LA_8_SemiSupervised_V15_2/Test_Part1"
# LOG_DIR="./result/VNet_Multi_V5/LA_8/Test"

# 数据集根路径
ROOT_PATH="./datasets/LA"  # 修改为你数据集的路径

# 类别数
NUM_CLASSES=2  # 修改为你的数据集类别数

# 输出数量
NUM_OUTPUT=2  # 假设这是默认的输出数量

# 测试保存路径 (如果没有，脚本中会忽略)
TEST_SAVE_PATH="None"  # 如果不保存结果，可以保持None

# 确保 LOG_DIR 存在，如果不存在则创建
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

# 遍历目录中的每个 .pth 文件
for MODEL_PATH in $PTH_DIR/*.pth; do
  # 从文件名中提取 epoch 编号 (可选)
  EPOCH=$(echo $MODEL_PATH | grep -oP '(?<=epoch_)\d+(?=_checkpoint.pth)')
  
  # 设置日志文件名以包含 epoch 编号
  LOG_FILE="$LOG_DIR/Test.log"

  echo "正在测试 $MODEL_PATH, 日志记录到 $LOG_FILE"

  # 使用当前的 .pth 文件调用 Python 脚本
  python -m Test_LA \
    --model_name "V15_Part1" \
    --model_load "$MODEL_PATH" \
    --log_path "$LOG_FILE" \
    --test_save_path "$TEST_SAVE_PATH" \
    --root_path "$ROOT_PATH" \
    --num_classes "$NUM_CLASSES" \
    --num_outputs "$NUM_OUTPUT"  # 假设这是默认的输出数量
done

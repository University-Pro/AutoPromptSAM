#!/bin/bash
# 配置参数
PTH_DIR="./result/VNet_Multi_V2/Amos_18/Pth"
LOG_DIR="./result/VNet_Multi_V2/Amos_18/Test"
ROOT_PATH="./datasets/Amos"
SCRIPT_NAME="Test_Amos.py"

# 检查Python环境
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        alias python=python3  # 兼容python3环境
    else
        echo "错误：未找到Python环境"
        exit 1
    fi
fi

# 创建日志目录
mkdir -p "$LOG_DIR"

# 批量测试
for MODEL_PATH in "${PTH_DIR}"/model_epoch_*_checkpoint.pth; do
    # 提取epoch数字
    EPOCH=$(basename "$MODEL_PATH" | grep -oE '[0-9]+')
    
    # 生成日志文件
    LOG_FILE="${LOG_DIR}/Test.log"
    
    echo "测试模型: $(basename "$MODEL_PATH")"
    echo "Epoch: ${EPOCH}"
    echo "日志文件: ${LOG_FILE}"
    
    # 执行测试命令
    if python -m "${SCRIPT_NAME%.py}" \
        --model_path "$MODEL_PATH" \
        --amos_data_path "$ROOT_PATH" \
        --output_dir "$LOG_DIR" \
        --split test \
        --metrics_log "$LOG_FILE"; then
        echo "[成功] Epoch ${EPOCH}"
    else
        echo "[失败] Epoch ${EPOCH}"
        # 显示错误日志（如果存在）
        if [ -f "$LOG_FILE" ]; then
            echo "=== 错误日志片段 ==="
            tail -n 20 "$LOG_FILE"
        else
            echo "未生成日志文件"
        fi
        exit 1
    fi
done

echo "所有模型测试完成！"
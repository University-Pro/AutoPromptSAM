"""
此脚本用于解析包含多个“最终统计”区块的日志文件，适用于Amos数据集
并筛选出具有最佳Dice（最高）和HD95（最低）指标的记录。
"""
import re
import argparse
from typing import List, Tuple, Optional

def parse_log_blocks(log_file: str) -> List[Tuple[str, float, float]]:
    """
    解析日志文件，提取每个“最终统计”区块的时间戳、Dice均值和HD95均值。

    :param log_file: 日志文件的路径。
    :return: 一个元组列表，每个元组包含 (时间戳, Dice均值, HD95均值)。
    """
    results = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # 使用“最终统计”作为分隔符，将日志分割成多个区块
        # re.split会保留分隔符前的部分，第一个元素通常是空的或无用的，所以我们从[1:]开始
        header_pattern = r"={10,}\s*最终统计\s*={10,}"
        blocks = re.split(header_pattern, content)

        if len(blocks) <= 1:
            print("警告: 在日志文件中未找到'最终统计'区块。")
            return []

        for block in blocks[1:]:
            # 定义正则表达式来捕获所需指标
            # 我们捕获每块中的第一个时间戳作为其唯一标识
            timestamp_match = re.search(r"\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]", block)
            dice_match = re.search(r"\[Dice\] 均值 ± 标准差:\s+([\d.]+)", block)
            hd95_match = re.search(r"\[HD95\] 均值 ± 标准差:\s+([\d.]+)", block)

            # 确保所有指标都被找到
            if timestamp_match and dice_match and hd95_match:
                timestamp = timestamp_match.group(1)
                dice_mean = float(dice_match.group(1))
                hd95_mean = float(hd95_match.group(1))
                
                results.append((timestamp, dice_mean, hd95_mean))

    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {log_file}")
    except Exception as e:
        print(f"解析日志时发生错误: {e}")
    
    return results

def main():
    """主函数，用于解析命令行参数并展示结果。"""
    parser = argparse.ArgumentParser(description="从日志文件中解析并筛选最佳分割指标")
    parser.add_argument("--log_path", type=str, required=True, help="需要解析的日志文件路径")
    args = parser.parse_args()

    all_results = parse_log_blocks(args.log_path)
    
    if not all_results:
        print("未从日志中解析出任何有效结果。")
        return

    # --- 打印所有解析出的结果 ---
    print("\n[所有解析结果]")
    print(f"{'时间戳':<22} | {'Dice (Mean)':>12} | {'HD95 (Mean)':>12}")
    print("-" * 52)
    for timestamp, dice, hd95 in all_results:
        print(f"{timestamp:<22} | {dice:>12.4f} | {hd95:>12.2f}")

    # --- 寻找并打印最佳结果 ---
    print("\n[最佳指标统计]")
    
    # Dice越高越好，所以用max
    best_dice_result = max(all_results, key=lambda x: x[1])
    
    # HD95越低越好，所以用min
    best_hd95_result = min(all_results, key=lambda x: x[2])
    
    print(f"最高 Dice: {best_dice_result[1]:.4f} (来自 {best_dice_result[0]})")
    print(f"最低 HD95: {best_hd95_result[2]:.2f} (来自 {best_hd95_result[0]})")

if __name__ == "__main__":
    main()
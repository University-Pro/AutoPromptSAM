"""
能够自动从新格式中的日志文件中查找出最好的指标
"""
import re
import argparse

def parse_log(log_file):
    """
    解析新格式日志文件，提取模型路径及对应的平均指标
    :param log_file: 日志文件路径
    :return: 包含模型路径、Dice、Jaccard、HD95、ASD的列表
    """
    results = []
    
    try:
        with open(log_file, 'r') as file:
            logs = file.read()

            # 匹配模型路径（支持多模型加载情况）
            model_pattern = r"Loading model weights from: (.+?\.pth)"
            model_paths = re.findall(model_pattern, logs)
            
            # 匹配最终平均指标（匹配带decoder编号和不带编号的两种格式）
            metric_pattern = r"Average metric (?:of decoder \d+): \[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]"
            metric_matches = re.findall(metric_pattern, logs)
            
            # 对齐模型路径和指标结果
            for i, (dice, jaccard, hd95, asd) in enumerate(metric_matches):
                # 取第一个模型路径（当有多个模型时按顺序对应）
                model_path = model_paths[i] if i < len(model_paths) else f"Unknown_Model_{i+1}"
                results.append((
                    model_path,
                    float(dice) * 100,    # Dice转为百分比
                    float(jaccard) * 100, # Jaccard转为百分比
                    float(hd95),          # HD95原始值
                    float(asd)            # ASD原始值
                ))

    except Exception as e:
        print(f"解析错误: {str(e)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="医学图像分割结果解析工具")
    parser.add_argument("--log_path", required=True, help="日志文件路径")
    args = parser.parse_args()

    results = parse_log(args.log_path)
    
    if not results:
        print("未找到有效结果")
        return

    # 打印详细结果
    print("\n[解析结果]")
    print(f"{'模型路径':<50} | {'Dice(%)':>8} | {'Jaccard(%)':>10} | {'HD95':>8} | {'ASD':>8}")
    print("-"*90)
    for path, dice, jaccard, hd95, asd in results:
        print(f"{path:<50} | {dice:>8.2f} | {jaccard:>10.2f} | {hd95:>8.2f} | {asd:>8.2f}")

    # 统计最佳结果
    if len(results) > 1:
        print("\n[最佳指标]")
        best_dice = max(results, key=lambda x: x[1])
        best_jaccard = max(results, key=lambda x: x[2])
        best_hd95 = min(results, key=lambda x: x[3])
        best_asd = min(results, key=lambda x: x[4])
        
        print(f"最高Dice: {best_dice[0]} -> {best_dice[1]:.2f}%")
        print(f"最高Jaccard: {best_jaccard[0]} -> {best_jaccard[2]:.2f}%")
        print(f"最低HD95: {best_hd95[0]} -> {best_hd95[3]:.2f}")
        print(f"最低ASD: {best_asd[0]} -> {best_asd[4]:.2f}")
    else:
        print("\n[最终结果]")
        print(f"Dice: {results[0][1]:.2f}% | Jaccard: {results[0][2]:.2f}% | HD95: {results[0][3]:.2f} | ASD: {results[0][4]:.2f}")

if __name__ == "__main__":
    main()
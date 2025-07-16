import re
import argparse
from typing import Dict, List, Tuple

def parse_log_blocks(log_file: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    解析日志文件，提取每个输出类型的统计指标。
    :param log_file: 日志文件的路径。
    :return: 一个字典，包含每个输出类型的Dice和HD95指标。
    """
    results = {}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 查找最终统计区块
        final_stats_pattern = r"={10,}\s*最终统计 \((.*?)\)\s*={10,}(.*?)(?=\n={10,}|$)"
        matches = re.findall(final_stats_pattern, content, re.DOTALL)
        
        if not matches:
            print("警告: 在日志文件中未找到'最终统计'区块。")
            return results
            
        for output_type, block_content in matches:
            dice_match = re.search(r"\[Dice\]\s+均值 ± 标准差:\s+([\d.]+)", block_content)
            hd95_match = re.search(r"\[HD95\]\s+均值 ± 标准差:\s+([\d.]+)", block_content)
            
            if dice_match and hd95_match:
                dice_mean = float(dice_match.group(1))
                hd95_mean = float(hd95_match.group(1))
                results[output_type] = {"Dice": dice_mean, "HD95": hd95_mean}
                
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {log_file}")
    except Exception as e:
        print(f"解析日志时发生错误: {e}")
    
    return results

def find_best_values(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    从结果中找出最佳值。
    :param results: 解析得到的结果字典。
    :return: 包含最佳值的字典。
    """
    best_values = {
        "Dice": {"value": -1.0, "output_type": ""},
        "HD95": {"value": float('inf'), "output_type": ""}
    }
    
    for output_type, metrics in results.items():
        # 更新最佳Dice (越大越好)
        if metrics["Dice"] > best_values["Dice"]["value"]:
            best_values["Dice"]["value"] = metrics["Dice"]
            best_values["Dice"]["output_type"] = output_type
            
        # 更新最佳HD95 (越小越好)
        if metrics["HD95"] < best_values["HD95"]["value"]:
            best_values["HD95"]["value"] = metrics["HD95"]
            best_values["HD95"]["output_type"] = output_type
    
    return best_values

def main():
    """主函数，用于解析命令行参数并展示结果。"""
    parser = argparse.ArgumentParser(description="从日志文件中解析并筛选最佳分割指标")
    parser.add_argument("--log_path", type=str, required=True, help="需要解析的日志文件路径")
    args = parser.parse_args()
    results = parse_log_blocks(args.log_path)
    
    if not results:
        print("未从日志中解析出任何有效结果。")
        return
    
    # 打印所有解析出的结果
    print("\n[所有输出类型的统计结果]")
    print(f"{'输出类型':<30} | {'Dice (Mean)':>12} | {'HD95 (Mean)':>12}")
    print("-" * 60)
    for output_type, metrics in results.items():
        print(f"{output_type:<30} | {metrics['Dice']:>12.4f} | {metrics['HD95']:>12.2f}")
    
    # 找出并打印最佳结果
    best_values = find_best_values(results)
    
    print("\n[最佳指标统计]")
    print(f"最高 Dice: {best_values['Dice']['value']:.4f} (来自 {best_values['Dice']['output_type']})")
    print(f"最低 HD95: {best_values['HD95']['value']:.2f} (来自 {best_values['HD95']['output_type']})")

if __name__ == "__main__":
    main()
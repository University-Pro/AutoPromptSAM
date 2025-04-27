import re
import argparse
from pathlib import Path

def parse_global_metrics(log_path):
    """
    解析最终统计信息
    返回字典包含：avg_dice, std_dice, avg_hd95, std_hd95
    """
    try:
        log_content = Path(log_path).read_text(encoding='utf-8')
    except Exception as e:
        raise ValueError(f"文件读取失败: {str(e)}")

    # 正则匹配最终统计指标
    metrics_pattern = r"""
    \[Dice\]\s均值\s±\s标准差:\s(\d+\.\d+)\s±\s(\d+\.\d+).*
    \[HD95\]\s均值\s±\s标准差:\s(\d+\.\d+)\s±\s(\d+\.\d+)
    """

    match = re.search(metrics_pattern, log_content, re.DOTALL|re.VERBOSE)
    
    if not match:
        raise ValueError("未找到全局统计信息")

    return {
        "avg_dice": float(match.group(1)),
        "std_dice": float(match.group(2)),
        "avg_hd95": float(match.group(3)),
        "std_hd95": float(match.group(4))
    }

def main(log_path):
    try:
        metrics = parse_global_metrics(log_path)
    except ValueError as e:
        print(f"错误: {str(e)}")
        return

    print("="*40 + " 全局统计 " + "="*40)
    print(f"Dice均值 ± 标准差: {metrics['avg_dice']:.4f} ± {metrics['std_dice']:.4f}")
    print(f"HD95均值 ± 标准差: {metrics['avg_hd95']:.2f} ± {metrics['std_hd95']:.2f}mm")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析医学影像全局统计指标')
    parser.add_argument('--log_file', type=str, required=True)
    args = parser.parse_args()
    main(args.log_file)
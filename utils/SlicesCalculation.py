from collections import defaultdict

def count_slices_from_file(file_path):
    """
    从 txt 文件读取文件名，统计每个 case 的切片数量，并按切片数量从高到低排序
    :param file_path: 存储文件名的 txt 文件路径
    """
    # 读取文件名
    with open(file_path, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]  # 去除空行和空格

    # 统计每个 case 的切片数量
    case_counts = defaultdict(int)
    for file_name in file_list:
        # 提取 case 标识符
        case_id = file_name.split("_slice")[0]
        case_counts[case_id] += 1

    # 按切片数量从高到低排序
    sorted_case_counts = sorted(case_counts.items(), key=lambda x: x[1], reverse=True)

    # 输出结果
    print("Case 切片统计 (从高到低排序):")
    for case, count in sorted_case_counts:
        print(f"{case}: {count} 切片")


if __name__ == "__main__":
    # 替换为你的 txt 文件路径
    # file_path = "datasets/ACDC/lists_ACDC/train.txt"  # 例如 ./datasets/train.txt
    file_path = "datasets/Synapse/list/train.txt"  # 例如./datasets/train.txt
    count_slices_from_file(file_path) 

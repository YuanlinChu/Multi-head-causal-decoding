import itertools
import os

def create_choices(inner_length=3, num_range=10, total_choices=None):
    """
    生成一个嵌套列表，每个内部列表包含指定长度的数字序列
    
    参数:
        inner_length (int): 内部列表的长度，默认为3
        num_range (int): 数字的范围，从0到num_range-1，默认为10（即0-9）
        total_choices (int): 要生成的选择总数，默认为None（生成所有可能组合）
    
    返回:
        list: 嵌套列表，形如[[0, 0, 0], [0, 0, 1], ...]
    """
    # 计算所有可能的组合
    all_combinations = list(itertools.product(range(num_range), repeat=inner_length))
    
    # 将元组转换为列表
    result = [list(combo) for combo in all_combinations]
    
    # 如果指定了总数，且小于所有可能的组合数，则截取指定数量
    if total_choices is not None and total_choices < len(result):
        result = result[:total_choices]
    
    return result


def save_choices_to_formatted_txt(choices, file_path):
    """
    将选择列表保存到TXT文件中，格式为[0, 0, 0], [0, 0, 1], [0, 0, 2]...
    
    参数:
        choices (list): 要保存的选择列表
        file_path (str): 保存的文件路径
    """
    # 确保目录存在（只有当文件路径包含目录时才创建目录）
    dir_name = os.path.dirname(file_path)
    if dir_name:  # 只有当目录名不为空时才创建目录
        os.makedirs(dir_name, exist_ok=True)
    
    # 修改格式化方式
    formatted_choices = ", ".join(str(choice) for choice in choices)
    
    # 将格式化后的字符串保存到文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_choices)
    
    print(f"已将选择列表保存到: {file_path}")


def main():
    # 生成默认参数的选择列表（3位数字，范围0-9）
    choices = create_choices()
    print(f"生成了 {len(choices)} 个选择")
    
    # 保存到格式化的TXT文件
    save_choices_to_formatted_txt(choices, "formatted_choices.txt")
    
if __name__ == "__main__":
    main()
import os

def get_folders_start_with_1(target_dir):
    """
    提取目标文件夹中所有以"1"开头的纯数字子文件夹名称，并转换为整型列表
    
    参数:
        target_dir: 目标文件夹的路径（相对路径或绝对路径）
    
    返回:
        list: 转换后的整型数字列表（按文件夹名称数字大小排序）
    """
    # 初始化结果列表
    result_list = []
    
    # 检查目标文件夹是否存在
    if not os.path.isdir(target_dir):
        print(f"错误：文件夹 '{target_dir}' 不存在！")
        return result_list
    
    # 遍历目标文件夹中的所有内容
    for item in os.listdir(target_dir):
        # 拼接完整路径
        item_path = os.path.join(target_dir, item)
        # 判断条件：是文件夹 + 名称以"1"开头 + 纯数字组成
        if os.path.isdir(item_path) and item.startswith("1") and item.isdigit():
            # 转换为整型并添加到列表
            result_list.append(int(item))
    
    # 对列表按数字大小排序（可选，根据需要决定是否保留）
    result_list.sort()
    
    return result_list

if __name__ == "__main__":
    # -------------------------- 配置区域 --------------------------
    # 请修改为你的目标文件夹路径（示例路径）
    TARGET_FOLDER = r"Z:\reflfluc2\Refl_Fluc_2025\High Frequency Reflectometry"  # <-- 请修改这里！
    # ----------------------------------------------------------------
    
    # 提取符合条件的文件夹名称并转换为整型列表
    folder_list = get_folders_start_with_1(TARGET_FOLDER)
    
    # 打印结果
    print("=" * 60)
    print(f"目标文件夹：{os.path.abspath(TARGET_FOLDER)}")
    print(f"以'1'开头的纯数字子文件夹数量：{len(folder_list)}")
    print(f"整型结果列表：{folder_list}")
    print(f"列表数据类型：{type(folder_list)}")
    print(f"列表元素类型：{type(folder_list[0]) if folder_list else '无'}")
    print("=" * 60)
    
    # 将结果保存为Python文件（可直接import使用）
    save_to_file = True  # 是否保存到文件
    if save_to_file:
        save_filename = "folder_int_list.py"
        with open(save_filename, "w", encoding="utf-8") as f:
            f.write(f"# 以'1'开头的纯数字子文件夹名称（整型列表）\n")
            f.write(f"# 生成时间：{os.popen('date').read().strip() if os.name != 'nt' else os.popen('date /t').read().strip()}\n")
            f.write(f"# 目标文件夹：{os.path.abspath(TARGET_FOLDER)}\n")
            f.write(f"folder_numbers = {folder_list}\n")
        print(f"\n结果已保存到文件：{os.path.abspath(save_filename)}")


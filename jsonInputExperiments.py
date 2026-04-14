import json
import os

# JSON 保存路径（当前目录下）
JSON_FILE = "experiment_time_records.json"

def load_existing_data():
    """加载已保存的历史数据（避免覆盖）"""
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("⚠️  现有 JSON 文件格式异常，将创建新文件")
                return {}
    return {}

def save_to_json(data):
    """保存数据到 JSON 文件（格式化，支持中文）"""
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 数据已保存到：{os.path.abspath(JSON_FILE)}")

def record_experiment():
    """循环录入实验编号和时间（起始/结束均手动输入）"""
    print("=" * 50)
    print("📝 实验时间录入工具（输入 'q' 或 'quit' 退出）")
    print("📌 时间支持小数格式（如 2.0、5.32、10.15）")
    print("=" * 50)
    
    experiment_data = load_existing_data()  # 加载历史记录

    while True:
        print("\n" + "-" * 30)
        # 1. 输入实验编号（必填，唯一标识）
        exp_id = input("请输入实验编号（如 EXP-001）：").strip()
        if exp_id.lower() in ["q", "quit"]:
            print("\n👋 退出录入，感谢使用！")
            break
        if not exp_id:
            print("❌ 实验编号不能为空，请重新输入！")
            continue

        # 2. 输入起始时间（支持小数）
        while True:
            start_time_input = input("请输入起始时间（如 2.0、5.32）：").strip()
            try:
                start_time = float(start_time_input)  # 转为数字（支持整数/小数）
                break
            except ValueError:
                print("❌ 时间格式错误！请输入数字（支持小数，如 3.14）")

        # 3. 输入结束时间（支持小数）
        while True:
            end_time_input = input("请输入结束时间（如 5.32、10.5）：").strip()
            try:
                end_time = float(end_time_input)
                # 简单校验：结束时间应大于起始时间
                if end_time <= start_time:
                    print("❌ 结束时间应大于起始时间，请重新输入！")
                    continue
                break
            except ValueError:
                print("❌ 时间格式错误！请输入数字（支持小数，如 6.89）")

        # 4. 存储数据（字段名改为 time_range，时间以列表形式保存）
        experiment_data[exp_id] = {
            "time_range": [start_time, end_time]  # 英文字段名
        }

        # 5. 实时保存
        save_to_json(experiment_data)
        print(f"🎉 实验 {exp_id} 记录成功！Time range：[{start_time}, {end_time}]")

if __name__ == "__main__":
    record_experiment()

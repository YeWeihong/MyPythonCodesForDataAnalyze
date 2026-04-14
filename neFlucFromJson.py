import json
import os
import matplotlib.pyplot as plt
from NeFluc import ReflAnalyzer  # 请确保这里导入正确（替换为实际模块名）

# ==================== 配置区 ====================
# JSON 文件路径（与脚本同目录，或改成绝对路径）
JSON_FILE = "experiment_time_records.json"

# 如果你想只测试某几发，可以在这里限制（留空表示全部）
# ONLY_THESE_SHOTS = ["158900", "157230"]
ONLY_THESE_SHOTS = []   # 空列表 = 画所有
# ================================================

def load_shot_configs(json_path):
    """读取 JSON 并返回 {shot_num_str: [t_start, t_end]} 的字典"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到配置文件: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确保 key 是字符串，方便后续处理
    configs = {}
    for shot_str, info in data.items():
        if "time_range" in info and len(info["time_range"]) == 2:
            configs[shot_str] = info["time_range"]
        else:
            print(f"Warning: shot {shot_str} 的 time_range 格式不正确，已跳过")
    
    return configs

def main():
    print("=== 开始批量绘制反射计 Judge=1 vs Judge=2 对比图 ===")
    
    configs = load_shot_configs(JSON_FILE)
    
    # 如果设置了只画部分炮，就过滤
    if ONLY_THESE_SHOTS:
        configs = {k: v for k, v in configs.items() if k in ONLY_THESE_SHOTS}
    
    print(f"共加载到 {len(configs)} 发炮的配置")
    
    for shot_str, (t_start, t_end) in configs.items():
        shot_num = int(shot_str)  # ReflAnalyzer 需要 int 类型
        print(f"\n正在处理 Shot #{shot_num}  [{t_start} - {t_end}] s")
        
        # Judge = 1 (Original)
        analyzer1 = ReflAnalyzer(shot_num, t_start, t_end, judge=1)
        t1, f1, psds1, freqs1 = analyzer1.run()
        
        # Judge = 2 (Amplitude Fluctuation)
        analyzer2 = ReflAnalyzer(shot_num, t_start, t_end, judge=2)
        t2, f2, psds2, freqs2 = analyzer2.run()
        
        if t1 is None or t2 is None or psds1 is None or psds2 is None:
            print(f"  → Shot #{shot_num} 数据读取失败，跳过")
            continue
        
        # 确保频率列表一致（正常情况下应该一样）
        # freqs1/freqs2 是 Python 列表，直接用 == 会返回布尔值而不是元素数组，
        # 因此用 != 来判断不等，或者使用 numpy.array_equal 做严格比较。
        try:
            import numpy as _np
            if not _np.array_equal(freqs1, freqs2):
                print(f"  → Shot #{shot_num} 两个 Judge 的频率列表不一致，使用 Judge=1 的频率")
                freqs2 = freqs1
        except Exception:
            if freqs1 != freqs2:
                print(f"  → Shot #{shot_num} 两个 Judge 的频率列表不一致，使用 Judge=1 的频率")
                freqs2 = freqs1
        
        # ================ 开始绘图 ================
        fig, axes = plt.subplots(4, 2, figsize=(7, 6), sharex=True, sharey=True)
        
        for i in range(4):
            freq_val = freqs1[i]  # GHz
            
            # 左：Original
            ax_left = axes[i, 0]
            title_l = f"#{shot_num} {freq_val:.3f} GHz (Original)"
            ReflAnalyzer.plot_on_axis(ax_left, t1, f1, psds1[i], title_l)
            
            # 右：Amp Fluc
            ax_right = axes[i, 1]
            title_r = f"#{shot_num} {freq_val:.3f} GHz (Amp Fluc)"
            ReflAnalyzer.plot_on_axis(ax_right, t2, f2, psds2[i], title_r)
        
        # X 轴标签只在最下面一行显示
        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        # 保存图片
        save_file = f"D:\FindEHO\Already\Compare_{shot_num}.png"
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
        plt.close(fig)  # 释放内存，防止批量时爆内存
        
        print(f"  → 图像已保存: {os.path.abspath(save_file)}")
    
    print("\n=== 所有炮处理完成 ===")

if __name__ == "__main__":
    main()
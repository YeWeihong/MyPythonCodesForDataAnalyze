# 保存为 manual_label_mds.py
import sys
sys.path.append(r"D:\MDSplusPy\python")
import MDSplus as mds
import matplotlib.pyplot as plt
import numpy as np

SERVER = '202.127.204.12'
TREE = 'efitrt_EAST'
SIGNAL = r'\WMHD'

def plot_shot_for_labeling(shotnu):
    """快速画一个放电的图，让你判断是否含模式"""
    try:
        c = mds.Connection(SERVER)
        c.openTree(TREE, shotnu)
        
        signal = c.get(SIGNAL).data()
        time = c.get(f'dim_of({SIGNAL})').data()
        
        # 画2秒到6秒的区间
        mask = (time >= 2.0) & (time <= 6.0)
        t = time[mask]
        s = signal[mask]
        
        plt.figure(figsize=(14, 6))
        
        # 上：原始信号
        plt.subplot(2, 1, 1)
        plt.plot(t, s, linewidth=1)
        plt.title(f'Shot {shotnu}: WMHD (MHD储能)', fontsize=14)
        plt.xlabel('时间 [s]')
        plt.ylabel('WMHD')
        plt.grid(True, alpha=0.3)
        
        # 下：时频谱（简单版，用短时傅里叶变换）
        plt.subplot(2, 1, 2)
        fs = 1.0 / (t[1] - t[2])  # 自动计算采样率
        f, t_spec, Sxx = signal.spectrogram(s, fs, nperseg=1024, noverlap=512)
        
        # 只画5-100kHz范围
        freq_mask = (f >= 5000) & (f <= 100000)
        plt.pcolormesh(t_spec, f[freq_mask]/1000, 10*np.log10(Sxx[freq_mask]), 
                       shading='gouraud', cmap='viridis')
        plt.colorbar(label='功率谱密度 [dB]')
        plt.title('时频谱 (5-100 kHz)')
        plt.xlabel('时间 [s]')
        plt.ylabel('频率 [kHz]')
        
        plt.tight_layout()
        plt.show()
        
        c.closeTree(TREE, shotnu)
        
        return True
        
    except Exception as e:
        print(f"Shot {shotnu} 读取失败: {e}")
        return False

# ================= 今晚执行 =================
if __name__ == '__main__':
    # 填入你最确定的5-10个放电号
    SUSPECTED_SHOTS = [159092, 159100, 159105, 159110, 159115]
    
    results = []
    
    for shotnu in SUSPECTED_SHOTS:
        print(f"\n{'='*50}")
        print(f"正在展示 Shot {shotnu}...")
        print(f"看图后回答下面问题")
        print(f"{'='*50}\n")
        
        success = plot_shot_for_labeling(shotnu)
        
        if not success:
            continue
        
        # 交互式输入
        has_mode = input(f"Shot {shotnu} 是否含有感兴趣的模式? (y/n): ")
        
        if has_mode.lower() == 'y':
            mode_type = input("模式类型? (ELM/BAE/TAE/GAM/其他): ")
            t_start = float(input("模式开始时间 [s]: "))
            t_end = float(input("模式结束时间 [s]: "))
            confidence = int(input("你的确认度 (1-5, 5=100%确定): "))
            
            results.append([shotnu, t_start, t_end, mode_type, confidence])
            print("✅ 已记录")
        else:
            print("❌ 跳过")
    
    # 保存标注结果
    if results:
        np.savetxt("rare_modes_manual.csv", results, fmt='%s', delimiter=',')
        print(f"\n✅ 已保存 {len(results)} 个标注到 rare_modes_manual.csv")
    else:
        print("\n❌ 没有记录任何模式")
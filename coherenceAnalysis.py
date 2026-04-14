import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import matplotlib.ticker as ticker

# 导入工具箱
from spectrum_toolbox import MDSDataLoader, ProbePlotter

class CrossPhaseAnalyzer:
    def __init__(self):
        self.loader = MDSDataLoader()
        self.plotter = ProbePlotter(self.loader)

    def get_aligned_data(self, shot, sig1_info, sig2_info, time_range, tree=None):
        """读取并对齐两个信号"""
        name1, num1 = sig1_info
        name2, num2 = sig2_info
        cfg1 = self.plotter.SIGNAL_CONFIG[name1]
        cfg2 = self.plotter.SIGNAL_CONFIG[name2]
        path1 = f"{cfg1['prefix']}{num1}{cfg1['suffix']}"
        path2 = f"{cfg2['prefix']}{num2}{cfg2['suffix']}"
        
        print(f"读取信号: {path1} & {path2}")
        t1, d1 = self.loader.get_signal(shot, path1, time_range, tree=tree)
        t2, d2 = self.loader.get_signal(shot, path2, time_range, tree=tree)
        
        # 去趋势
        d1 = signal.detrend(d1)
        d2 = signal.detrend(d2)

        # 对齐 (以采样率高的为准，或者如果接近则以 t1 为准)
        if len(t1) != len(t2):
             d2_interp = np.interp(t1, t2, d2)
             return t1, d1, d2_interp, path1, path2
        else:
             return t1, d1, d2, path1, path2

    def plot_cross_spectrum(self, shot, sig1_info, sig2_info, 
                            time_range, freq_range=[0, 10], 
                            nperseg=2048, smooth_width=15, 
                            coh_threshold=0.3, tree=None):
        """
        绘制 相干性谱(上) 和 相位差谱(下)
        
        Parameters:
        - coh_threshold: 相位图的掩膜阈值。只有相干性 > 该值的点才显示相位颜色。
        """
        
        # 1. 获取数据
        t, x, y, label1, label2 = self.get_aligned_data(shot, sig1_info, sig2_info, time_range, tree)
        fs = 1 / np.mean(np.diff(t))
        
        # 2. STFT 计算
        # Pxy 是复数，包含了幅值(相干性来源)和角度(相位差来源)
        f, t_spec, Zxx = signal.stft(x, fs, window='hann', nperseg=nperseg, noverlap=nperseg//2)
        _, _, Zyy = signal.stft(y, fs, window='hann', nperseg=nperseg, noverlap=nperseg//2)

        Pxx = np.abs(Zxx)**2
        Pyy = np.abs(Zyy)**2
        Pxy = Zxx * np.conj(Zyy) # 注意：这里决定了相位差是 (Sig1 - Sig2) 还是反过来

        # 3. 时间平滑 (Smoothing)
        # 这一步对于消除相位图的随机噪点至关重要
        Pxx_smooth = ndimage.uniform_filter1d(Pxx, size=smooth_width, axis=-1, mode='nearest')
        Pyy_smooth = ndimage.uniform_filter1d(Pyy, size=smooth_width, axis=-1, mode='nearest')
        Pxy_smooth = ndimage.uniform_filter1d(Pxy, size=smooth_width, axis=-1, mode='nearest')

        # 4. 计算相干性
        Coh = (np.abs(Pxy_smooth)**2) / (Pxx_smooth * Pyy_smooth + 1e-10)
        
        # 5. 计算相位差 (弧度)
        # np.angle 返回范围 [-pi, pi]
        Phase = np.angle(Pxy_smooth)

        # 6. 数据切片 (Freq Range)
        f_khz = f / 1000.0
        t_abs = t_spec + time_range[0]
        
        if freq_range:
            f_mask = (f_khz >= freq_range[0]) & (f_khz <= freq_range[1])
            plot_coh = Coh[f_mask, :]
            plot_phase = Phase[f_mask, :]
            plot_f = f_khz[f_mask]
            extent = [t_abs[0], t_abs[-1], plot_f[0], plot_f[-1]]
        else:
            plot_coh = Coh
            plot_phase = Phase
            extent = [t_abs[0], t_abs[-1], f_khz[0], f_khz[-1]]

        # 7. 应用掩膜 (Masking)
        # 创建一个 masked array，将相干性低的地方的相位数据隐藏
        plot_phase_masked = np.ma.masked_where(plot_coh < coh_threshold, plot_phase)

        # ================= 绘图 =================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
        
        # --- 子图1: 相干性 ---
        im1 = ax1.imshow(plot_coh, aspect='auto', origin='lower', 
                         cmap='jet', extent=extent, vmin=0, vmax=1)
        cb1 = plt.colorbar(im1, ax=ax1, pad=0.01)
        cb1.set_label(r"Coherence $\gamma^2$")
        
        ax1.set_ylabel("Frequency [kHz]")
        ax1.set_title(f"Cross-Spectrum Analysis: #{shot} | {label1} vs {label2}")
        ax1.grid(False)

        # --- 子图2: 相位差 ---
        # 使用 'twilight' 或 'hsv' 这种循环色谱，因为 pi 和 -pi 物理上是一样的
        im2 = ax2.imshow(plot_phase_masked, aspect='auto', origin='lower', 
                         cmap='twilight', extent=extent, vmin=-np.pi, vmax=np.pi)
        
        # 设置相位图背景色（被Mask掉的部分显示为灰色）
        ax2.set_facecolor('#e0e0e0') 
        
        cb2 = plt.colorbar(im2, ax=ax2, pad=0.01)
        cb2.set_label(r"Phase Diff [rad]")
        
        # 将 Colorbar 刻度设置为 pi 的倍数，看起来更专业
        cb2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb2.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

        ax2.set_ylabel("Frequency [kHz]")
        ax2.set_xlabel("Time [s]")
        ax2.grid(False)

        print(f"绘图完成. Mask阈值: Coherence < {coh_threshold}")
        return fig

# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    # 1. 修改参数
    SHOT = 137569
    TIME_RANGE = [9.5, 13.0]
    FREQ_RANGE = [0, 5]  # kHz
    
    # 信号
    SIG1 = ('khpt', 7)    # 参考信号
    SIG2 = ('ece1-9', 8)  # 目标信号

    analyzer = CrossPhaseAnalyzer()

    try:
        fig = analyzer.plot_cross_spectrum(
            shot=SHOT,
            sig1_info=SIG1,
            sig2_info=SIG2,
            time_range=TIME_RANGE,
            freq_range=FREQ_RANGE,
            nperseg=512*64,      # 保持高频率分辨率
            smooth_width=21,   # 保持平滑以获得干净的相位
            coh_threshold=0.4  # 关键参数：低于0.4相关性的地方，不显示相位颜色（显示灰色背景）
        )
        
        plt.savefig(f"CrossPhase_{SHOT}.png", dpi=150)
        plt.show()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
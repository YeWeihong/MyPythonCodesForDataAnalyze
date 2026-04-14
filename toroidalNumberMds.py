import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# 导入工具箱
from spectrum_toolbox import MDSDataLoader, ProbePlotter

class ToroidalModeAnalyzerPro:
    def __init__(self):
        self.loader = MDSDataLoader()
        self.plotter = ProbePlotter(self.loader)
        
        # 动态配置检测
        if 'lhpt' not in self.plotter.SIGNAL_CONFIG:
            self.plotter.SIGNAL_CONFIG['lhpt'] = {'prefix': '\\lhp', 'suffix': 't', 'unit': 'T'}

    def get_aligned_triplet(self, shot, sig_k, sig_l, sig_ref, time_range, tree=None):
        """读取并对齐三个信号"""
        def load_one(sig_info):
            name, num = sig_info
            cfg = self.plotter.SIGNAL_CONFIG[name]
            path = f"{cfg['prefix']}{num}{cfg['suffix']}"
            t, d = self.loader.get_signal(shot, path, time_range, tree=tree)
            return t, signal.detrend(d), path

        print("正在读取信号组...")
        t_k, d_k, path_k = load_one(sig_k)
        t_l, d_l, path_l = load_one(sig_l)
        t_r, d_r, path_r = load_one(sig_ref)
        
        # 统一插值到 t_k
        d_l_interp = np.interp(t_k, t_l, d_l)
        d_r_interp = np.interp(t_k, t_r, d_r)
        
        return t_k, d_k, d_l_interp, d_r_interp, path_k, path_l, path_r

    def compute_comprehensive_analysis(self, shot, 
                                       sig_khp, sig_lhp, sig_ref, 
                                       probe_angle_deg, 
                                       time_range, freq_range=[0, 10], 
                                       nperseg=2048, smooth_width=15, 
                                       coh_threshold=0.4, tree=None):
        
        # 1. 获取数据
        t, x_k, x_l, x_ref, lab_k, lab_l, lab_ref = self.get_aligned_triplet(
            shot, sig_khp, sig_lhp, sig_ref, time_range, tree
        )
        fs = 1 / np.mean(np.diff(t))
        
        # 2. STFT 计算
        window = 'hann'
        overlap = nperseg // 2
        f, t_spec, Z_k = signal.stft(x_k, fs, window=window, nperseg=nperseg, noverlap=overlap)
        _, _, Z_l = signal.stft(x_l, fs, window=window, nperseg=nperseg, noverlap=overlap)
        _, _, Z_ref = signal.stft(x_ref, fs, window=window, nperseg=nperseg, noverlap=overlap)

        # 3. 计算功率谱与互功率谱
        def smooth(d): return ndimage.uniform_filter1d(d, size=smooth_width, axis=-1, mode='nearest')
        
        # 自功率谱 (用于分母)
        P_kk_s = smooth(np.abs(Z_k)**2)
        P_ll_s = smooth(np.abs(Z_l)**2)
        P_rr_s = smooth(np.abs(Z_ref)**2)
        
        # 互功率谱 (含相位信息)
        P_kr_s = smooth(Z_k * np.conj(Z_ref))  # K vs Ref
        P_lr_s = smooth(Z_l * np.conj(Z_ref))  # L vs Ref
        P_kl_s = smooth(Z_k * np.conj(Z_l))    # K vs L (用于互检)

        # 4. 计算相关性 (Coherence)
        eps = 1e-10
        Coh_kr = (np.abs(P_kr_s)**2) / (P_kk_s * P_rr_s + eps)
        Coh_lr = (np.abs(P_lr_s)**2) / (P_ll_s * P_rr_s + eps)
        
        # 5. 计算相位与模数
        phi_k = np.angle(P_kr_s)
        phi_l = np.angle(P_lr_s)
        
        # 相位差相减: (Phi_K - Phi_Ref) - (Phi_L - Phi_Ref) = Phi_K - Phi_L
        complex_diff = np.exp(1j * phi_k) * np.exp(-1j * phi_l)
        delta_phi = np.angle(complex_diff)
        
        # 原始模数 (Continuous)
        n_raw = delta_phi / np.deg2rad(probe_angle_deg)

        # 6. 数据切片 (Freq Range)
        f_khz = f / 1000.0
        t_abs = t_spec + time_range[0]
        
        if freq_range:
            mask = (f_khz >= freq_range[0]) & (f_khz <= freq_range[1])
            f_plot = f_khz[mask]
            # 切片所有数据
            Coh_kr_plot = Coh_kr[mask, :]
            Coh_lr_plot = Coh_lr[mask, :]
            n_raw_plot = n_raw[mask, :]
            phi_k_plot = phi_k[mask, :]
            phi_l_plot = phi_l[mask, :]
        else:
            f_plot = f_khz
            Coh_kr_plot, Coh_lr_plot, n_raw_plot = Coh_kr, Coh_lr, n_raw
            phi_k_plot, phi_l_plot = phi_k, phi_l

        extent = [t_abs[0], t_abs[-1], f_plot[0], f_plot[-1]]
        
        # 掩膜处理 (用于 n 的显示)
        # 只有当两个信号与 Reference 的相关性都较高时，计算出的 n 才可信
        combined_coh = np.minimum(Coh_kr_plot, Coh_lr_plot)
        n_masked = np.ma.masked_where(combined_coh < coh_threshold, n_raw_plot)

        # ============================================================
        # 图表 1: 时频分布全景图 (Spectrogram Dashboard)
        # ============================================================
        fig1 = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.15)
        
        # --- 子图1: KHP vs Ref 相关性 ---
        ax1 = fig1.add_subplot(gs[0, 0])
        im1 = ax1.imshow(Coh_kr_plot, aspect='auto', origin='lower', extent=extent, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax1, label='Coherence')
        ax1.set_title(f"Coherence: {lab_k} vs {lab_ref}")
        ax1.set_ylabel("Freq [kHz]")
        ax1.set_xticklabels([]) # 隐藏x轴标尺

        # --- 子图2: LHP vs Ref 相关性 ---
        ax2 = fig1.add_subplot(gs[0, 1])
        im2 = ax2.imshow(Coh_lr_plot, aspect='auto', origin='lower', extent=extent, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, label='Coherence')
        ax2.set_title(f"Coherence: {lab_l} vs {lab_ref}")
        ax2.set_yticklabels([]) # 隐藏y轴标尺
        ax2.set_xticklabels([])

        # --- 子图3: 原始模数 n (Continuous) ---
        ax3 = fig1.add_subplot(gs[1, 0])
        # 使用 RdBu 色谱，0为白色，红正蓝负，范围设为 -5 到 5
        im3 = ax3.imshow(n_masked, aspect='auto', origin='lower', extent=extent, cmap='RdYlBu_r', vmin=-5, vmax=5)
        plt.colorbar(im3, ax=ax3, label='Raw n value')
        ax3.set_title(f"Raw Toroidal Mode $n$ (Float)")
        ax3.set_ylabel("Freq [kHz]")
        ax3.set_xlabel("Time [s]")
        ax3.set_facecolor('#e0e0e0') # 灰色背景

        # --- 子图4: 整数模数 n (Discrete) ---
        ax4 = fig1.add_subplot(gs[1, 1])
        # 离散色谱
        bounds = np.arange(-5.5, 6.5, 1)
        cmap_disc = plt.cm.get_cmap('RdYlBu_r', len(bounds)-1)
        norm_disc = mcolors.BoundaryNorm(bounds, cmap_disc.N)
        
        im4 = ax4.imshow(n_masked, aspect='auto', origin='lower', extent=extent, cmap=cmap_disc, norm=norm_disc)
        cb4 = plt.colorbar(im4, ax=ax4, ticks=np.arange(-5, 6))
        cb4.set_label('Integer n')
        ax4.set_title(f"Discrete Mode $n$ (Rounded)")
        ax4.set_xlabel("Time [s]")
        ax4.set_yticklabels([])
        ax4.set_facecolor('#e0e0e0')

        fig1.suptitle(f"Toroidal Mode Analysis Dashboard #{shot}\nRef: {lab_ref}, DeltaAngle: {probe_angle_deg}°", fontsize=14)

        # ============================================================
        # 图表 2: 频域特性分析图 (Time-Averaged Profiles)
        # ============================================================
        fig2, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        
        # 对时间轴求平均 (axis=1)
        coh_kr_mean = np.mean(Coh_kr_plot, axis=1)
        coh_lr_mean = np.mean(Coh_lr_plot, axis=1)
        # 原始n的平均 (注意：Mask掉的地方不参与平均，或者全算)
        # 这里使用加权平均，权重为相关性，这样得到的 n 更准
        n_mean = np.sum(n_raw_plot * combined_coh, axis=1) / (np.sum(combined_coh, axis=1) + eps)
        
        # 计算相位差平均 (需注意周期性，这里简单展示)
        # 更严谨的做法是矢量平均
        
        # --- Plot 1: Coherence vs Freq ---
        axs[0].plot(f_plot, coh_kr_mean, label=f'{lab_k}-{lab_ref}', color='b')
        axs[0].plot(f_plot, coh_lr_mean, label=f'{lab_l}-{lab_ref}', color='r')
        axs[0].set_title("Avg Coherence Spectrum")
        axs[0].set_xlabel("Frequency [kHz]")
        axs[0].set_ylabel("Coherence")
        axs[0].set_ylim(0, 1.05)
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # --- Plot 2: Phase Difference vs Freq (vs Ref) ---
        # 这里展示两个探针分别相对于 Ref 的相位
        # 仅展示高相关性的点
        valid_idx = combined_coh.mean(axis=1) > 0.3
        axs[1].plot(f_plot[valid_idx], np.angle(np.mean(np.exp(1j*phi_k_plot), axis=1))[valid_idx], '.', label=f'{lab_k}', markersize=2)
        axs[1].plot(f_plot[valid_idx], np.angle(np.mean(np.exp(1j*phi_l_plot), axis=1))[valid_idx], '.', label=f'{lab_l}', markersize=2)
        axs[1].set_title("Avg Phase vs Ref (Coh>0.3)")
        axs[1].set_xlabel("Frequency [kHz]")
        axs[1].set_ylabel("Phase [rad]")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        # --- Plot 3: Toroidal Mode n vs Freq ---
        # 绘制 Raw n
        axs[2].plot(f_plot, n_mean, 'k-', linewidth=1.5, label='Avg Raw n')
        # 绘制散点，颜色代表相关性强弱
        sc = axs[2].scatter(f_plot, n_mean, c=combined_coh.mean(axis=1), cmap='viridis', s=20, zorder=5)
        plt.colorbar(sc, ax=axs[2], label='Avg Coherence')
        
        axs[2].set_title("Avg Toroidal Mode Number")
        axs[2].set_xlabel("Frequency [kHz]")
        axs[2].set_ylabel("Mode Number n")
        axs[2].set_ylim(-5, 5)
        axs[2].grid(True, alpha=0.3)
        axs[2].axhline(1, color='r', linestyle='--', alpha=0.3)
        axs[2].axhline(2, color='r', linestyle='--', alpha=0.3)
        
        return fig1, fig2

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    SHOT = 73895
    TIME_RANGE = [17.0, 19.0]
    FREQ_RANGE = [0, 10] # kHz
    DELTA_ANGLE = 22.029  

    SIG_K = ('khpt', 7) 
    SIG_L = ('lhpt', 7) 
    SIG_REF = ('ece10-48', 13) 

    analyzer = ToroidalModeAnalyzerPro()

    try:
        print(f"开始全方位分析: #{SHOT} ...")
        fig_maps, fig_profiles = analyzer.compute_comprehensive_analysis(
            shot=SHOT,
            sig_khp=SIG_K,
            sig_lhp=SIG_L,
            sig_ref=SIG_REF,
            probe_angle_deg=DELTA_ANGLE,
            time_range=TIME_RANGE,
            freq_range=FREQ_RANGE,
            nperseg=2048*16,   # 提高频率分辨率
            smooth_width=31,  # 平滑宽度
            coh_threshold=0.5 # 掩膜阈值
        )
        
        # 保存图片
        fig_maps.savefig(f"Mode_Maps_{SHOT}.png", dpi=150, bbox_inches='tight')
        fig_profiles.savefig(f"Mode_Profiles_{SHOT}.png", dpi=150, bbox_inches='tight')
        
        print("分析完成，图片已保存。")
        plt.show()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
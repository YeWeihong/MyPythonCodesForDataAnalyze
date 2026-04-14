# NeFluc.py
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import struct
from scipy.fft import fft, ifft, fftshift, ifftshift

class ReflAnalyzer:
    def __init__(self, shot, time_start, time_end, judge=2, base_path=r'D:\FindEHO'):
        """
        初始化反射仪分析器
        :param shot: 炮号 (int)
        :param time_start: 开始时间 (s)
        :param time_end: 结束时间 (s)
        :param judge: 1=原始信号(High Res), 2=幅度信号(Low Res/Filtered)
        :param base_path: 本地保存图片的根目录
        """
        self.shot = int(shot)
        self.shot_str = str(shot)
        self.time_span = [time_start, time_end]
        self.judge = judge
        self.base_path = base_path
        
        # 先给默认值，后面根据 card_name 自动修正
        self.fs = 2 * 10**6
        self.time_delay = 6.993 * 10**-3
        
        # 2025年所有可能的根目录（以后再乱直接加在这里）
        self.ROOTS_2025 = [
            r'Z:\reflfluc2\Refl_Fluc_2025\High Frequency Reflectometry',
            r'Z:\reflfluc2\Refl_Fluc_2025\Fluc',
        ]

        # 初始化路径配置
        self._setup_config()
        self._update_fs_by_card()

    def _update_fs_by_card(self):
        """根据板卡名称和炮号自动设置采样率"""
        if self.card_name == "Doppler":
            if 79651 <= self.shot <= 80495:
                self.fs = 20 * 10**6
            elif 80496 <= self.shot <= 97534:
                self.fs = 10 * 10**6
            else:
                self.fs = 20 * 10**6
        else:
            self.fs = 2 * 10**6
            
    def _setup_config(self):
        """统一配置路径和板卡信息，2025年及以后全部走自动搜索"""
        s = self.shot

        # ============ 2025年及以后：全部走自动搜索路径 ============
        if s >= 149000:
            self.data_root = None                  # 标记为自动搜索
            self.card_name = 'O_P2'
            self.save_folder = '2025ECM2_4'         # 你可以改成 2025ECM2_4HF/LF，随你喜欢
            return                                 # 直接返回！不要继续往下走 old elif

        # ============ 以下是老炮号（<149000）的硬编码区间 ============
        if 76181 <= s <= 80699:
            self.data_root = r'Z:\reflfluc2\Refl_Fluc_2018\O_20-40'
            self.save_folder = '2018ECM2_4'
            self.card_name = '5105A'
        elif 80710 <= s <= 81692:
            self.data_root = r'Z:\reflfluc2\Refl_Fluc_2018\X_W'
            self.save_folder = '2018ECM2_4'
            self.card_name = '5105A'
        elif 83218 <= s <= 94686:
            self.data_root = r'Z:\reflfluc2\Refl_Fluc_2019'
            self.save_folder = '2019ECM2_4'
            self.card_name = 'O_P1'
        elif 94688 <= s <= 97422:
            self.data_root = r'Z:\reflfluc2\Refl_Fluc_2020'
            self.save_folder = '2020ECM2_4'
            self.card_name = 'O_P1'
        elif 97423 <= s <= 107851:
            self.data_root = r'Z:\reflfluc2\Refl_Fluc_2021'
            self.save_folder = '2021ECM2_4'
            self.card_name = 'O_P1'
        elif 107960 <= s <= 120623:
            self.data_root = r'Z:\reflfluc2\Refl_Fluc_2022'
            self.save_folder = '2022ECM2_4'
            self.card_name = 'O_P2'
        elif 121675 <= s <= 137931:
            self.data_root = r'Z:\reflfluc2\Refl_Fluc_2023'
            self.save_folder = '2023ECM2_4'
            self.card_name = 'O_P2'
        elif 132043 <= s <= 148211:
            self.data_root = r'Z:\reflfluc2\Refl_Fluc_2024'
            self.save_folder = '2024ECM2_4'
            self.card_name = 'O_P2'
        else:
            # 只有真正 <149000 且不在任何区间，才走到这里（基本不会发生）
            print(f"Warning: Shot {s} 不在任何已知区间，使用2024作为默认")
            self.data_root = r'Z:\reflfluc2\Refl_Fluc_2024'
            self.save_folder = '2024ECM2_4'
            self.card_name = 'O_P2'

        # 老炮号设置完后统一创建 save_path
        self.save_path = os.path.join(self.base_path, self.save_folder)

    def _find_real_bin_file(self):
        """寻找真实存在的 .bin 文件"""
        # 2025年及以后：走自动搜索
        if self.shot >= 149000 or self.data_root is None:
            # 自动搜索逻辑（和之前一样）
            for root in self.ROOTS_2025:
                folder = os.path.join(root, self.shot_str)
                if not os.path.isdir(folder):
                    continue
                candidates = [
                    os.path.join(folder, "O_P2.bin"),
                    os.path.join(folder, "o_p2.bin"),
                    os.path.join(folder, f"{self.shot_str}_O_P2.bin"),
                    os.path.join(folder, "O_P2.BIN"),
                ]
                for p in candidates:
                    if os.path.exists(p):
                        print(f"  → Shot #{self.shot} 数据定位成功: {p}")
                        return p
            # 递归保底搜索...
            # （同之前代码）

        # 老炮号：走固定路径
        else:
            path = os.path.join(self.data_root, self.shot_str, f"{self.card_name}.bin")
            if os.path.exists(path):
                return path
            # 兼容性再试一次 O_P2.bin
            path2 = os.path.join(self.data_root, self.shot_str, "O_P2.bin")
            return path2 if os.path.exists(path2) else None

    def _get_card_params(self):
        """获取频率配置和IQ通道映射"""
        cn = self.card_name
        # 通道映射 [Row1, Row2] -> Python Index (MATLAB index - 1)
        # IQ Sign
        # Frequencies
        
        if cn in ['O_P1', 'O_P2', '5105A', '5105B', '5105C']:
            rows = [[4, 5], [0, 1], [2, 3], [6, 7]]
            iq_sign = [1, 1, -1, -1]
            freqs = [20.4, 24.8, 33, 40]
        elif cn in ['U_P1']:
            rows = [[4, 5], [0, 1], [2, 3], [6, 7]]
            iq_sign = [1, 1, 1, 1]
            freqs = [42.4, 48, 52.6, 57.2]
        elif cn in ['U_P2']:
            rows = [[4, 5], [0, 1], [2, 3], [6, 7]]
            iq_sign = [1, 1, -1, -1]
            freqs = [42.4, 48, 52.6, 57.2]
        elif cn in ['V_P1', 'V_P2']:
            rows = [[0, 1], [2, 3], [4, 5], [6, 7]]
            iq_sign = [-1, -1, -1, -1]
            freqs = [61.2, 65.6, 69.2, 73.6]
        elif cn in ['W_P1']:
            rows = [[6, 7], [2, 3], [0, 1], [4, 5]]
            iq_sign = [1, 1, 1, 1]
            freqs = [79.2, 85.2, 91.8, 96]
        elif cn in ['W_P2', 'Doppler']:
            rows = [[6, 7], [2, 3], [0, 1], [4, 5]]
            iq_sign = [1, 1, -1, -1]
            freqs = [79.2, 85.2, 91.8, 96] if cn == 'W_P2' else [56, 61, 66, 70]
        else:
            raise ValueError(f"Unknown card name: {cn}")

        # 构建完整文件名
        # 如果 data_root 为 None（标记为需要自动搜索），则使用 _find_real_bin_file
        if getattr(self, 'data_root', None) is None:
            filename = self._find_real_bin_file()
            if filename is None:
                raise FileNotFoundError(f"无法定位 Shot {self.shot} 的 .bin 文件（自动搜索失败）")
        else:
            filename = os.path.join(self.data_root, self.shot_str, f"{cn}.bin")
            # 如果找不到特定文件，尝试通用名 (如原代码中的逻辑)
            if not os.path.exists(filename) and cn in ['5105A', '5105B', '5105C', 'Doppler']:
                # 有时候文件名就是卡名，保持原样
                pass

        return filename, rows, iq_sign, freqs

    # ==================== 静态计算辅助函数 ====================
    
    @staticmethod
    def fft_bandpass(data, band_pass, fs):
        """频域带通滤波"""
        band_pass = sorted(band_pass)
        L = len(data)
        y_data = fft(data)
        # 频率轴 (kHz)
        f = (np.arange(L) * (fs / L) - fs / 2) / 1000 
        y_data = fftshift(y_data)
        
        # 置零
        mask = (f < band_pass[0]) | (f > band_pass[1])
        y_data[mask] = 0
        
        y_data = ifftshift(y_data)
        return ifft(y_data)

    @staticmethod
    def refl_average(time_arr, data, L_avg):
        """降采样平均"""
        if L_avg <= 1: return time_arr, data
        num_blocks = len(data) // L_avg
        keep_len = num_blocks * L_avg
        
        d_reshape = data[:keep_len].reshape((num_blocks, L_avg))
        t_reshape = time_arr[:keep_len].reshape((num_blocks, L_avg))
        
        return np.mean(t_reshape, axis=1), np.mean(d_reshape, axis=1)

    def _process_amplitude(self, t_in, signal_in):
        """对应 MATLAB refl_amp 函数"""
        band_pass = getattr(self, 'band_pass', [-1000, -600])   # kHz
        average_point = getattr(self, 'average_point', 4)

        s_filtered = self.fft_bandpass(signal_in, band_pass, self.fs)
        s_abs = np.abs(s_filtered)
        return self.refl_average(t_in, s_abs, average_point)

    @staticmethod
    def psd_me(x, fs, fftpoint):
        """功率谱密度计算"""
        L_x = len(x)
        num_segments = int(L_x // fftpoint)
        if num_segments == 0: 
            # 数据不足以进行一次FFT
            return np.zeros(fftpoint), np.zeros(fftpoint)
            
        m = 2 * num_segments - 1
        y = np.zeros((fftpoint, m), dtype=x.dtype)
        
        for k in range(m):
            idx = int(k * fftpoint / 2)
            y[:, k] = x[idx : idx + fftpoint]
            
        y_fft = fft(y, axis=0) / fftpoint
        y_psd = np.mean(np.abs(y_fft)**2, axis=1)
        
        P = fftshift(y_psd)
        f = np.arange(-fftpoint/2, fftpoint/2) * (fs / fftpoint)
        return P, f

    # ==================== 核心运行函数 ====================

    def run(self):
        self._update_fs_by_card()
        """
        读取数据并计算 PSD
        Returns:
            time_array (1D), freq_array (1D), PSD_list (List of 4 2D arrays), freqs_label (List)
        """
        try:
            filename, rows, iq_sign, freqs = self._get_card_params()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None, None, None, None

        if not filename or not isinstance(filename, str) or not os.path.exists(filename):
            print(f"Error: File does not exist: {filename}")
            return None, None, None, None

        channel_cnt = 8
        t1, t2 = self.time_span[0], self.time_span[1]
        L_time = 0.5 # 每次读取 0.5s 数据以防内存溢出
        
        time_chunks = np.arange(t1, t2, L_time)
        if time_chunks[-1] < t2:
            time_chunks = np.append(time_chunks, t2)
            
        # 结果容器
        all_t = []

        # 默认计算 4 个通道；如果外部指定了 target_channels，就只算这些通道
        target_channels = getattr(self, 'target_channels', [0, 1, 2, 3])
        if isinstance(target_channels, int):
            target_channels = [target_channels]
        target_channels = sorted(set(target_channels))

        all_PSD = {ch: [] for ch in target_channels}
        final_f = None
        
        print(f"Processing Shot {self.shot} | Judge={self.judge} | {self.card_name}")
        
        with open(filename, 'rb') as fid:
            for i in range(len(time_chunks) - 1):
                t_start_chunk = time_chunks[i]
                t_end_chunk = time_chunks[i+1]
                
                # 计算读取位置
                read_time = t_start_chunk - self.time_delay
                offset_bytes = int(channel_cnt * self.fs * read_time * 2)
                samples_to_read = int(np.ceil((t_end_chunk - t_start_chunk) * self.fs))
                
                # 读取
                fid.seek(offset_bytes)
                raw = np.fromfile(fid, dtype=np.int16, count=samples_to_read * channel_cnt)
                if len(raw) == 0: break
                
                # Reshape (Fortran Order for MATLAB compatibility)
                data_chunk = raw.reshape((channel_cnt, -1), order='F')
                
                # IQ 合成
                Signals = []
                for ch in range(4):
                    r1, r2 = rows[ch]
                    # S = I + sign*j*Q
                    sig = data_chunk[r1, :] + iq_sign[ch] * 1j * data_chunk[r2, :]
                    Signals.append(sig)
                
                # 生成时间轴
                t_temp = np.linspace(t_start_chunk, t_end_chunk, data_chunk.shape[1])
                
                # --- Judge 分支处理 ---
                proc_signals = []
                proc_t = []
                
                if self.judge == 1:
                    # 原始信号
                    # fftpoint = 512 * 4
                    proc_signals = Signals
                    proc_t = t_temp
                    fs_temp = self.fs
                else: # judge == 2
                    # 幅度信号处理
                    # fftpoint = 512
                    for sig in Signals:
                        t_out, s_out = self._process_amplitude(t_temp, sig)
                        proc_signals.append(s_out)
                    proc_t = t_out
                    # 计算降采样后的fs
                    fs_temp = 1.0 / (proc_t[1] - proc_t[0]) if len(proc_t) > 1 else self.fs/4
                
                # --- 计算 PSD (滑动窗口) ---
                if self.judge == 1:
                    fftpoint = getattr(self, 'fftpoint_j1', 2048)

                    if self.card_name == "Doppler":
                        step_factor = getattr(self, 'step_factor_j1_doppler', 2)
                    else:
                        step_factor = getattr(self, 'step_factor_j1', 8)

                elif self.judge == 2:
                    fftpoint = getattr(self, 'fftpoint_j2', 512)
                    step_factor = getattr(self, 'step_factor_j2', 4)

                else:
                    raise ValueError(f"Unsupported judge: {self.judge}")
                g = fftpoint                        # 现在窗口长度 = fftpoint（原来是 5/4，这里简化）
                step_samples = fftpoint // step_factor   # 关键：步长大幅减小
                L_sig = len(proc_signals[0])
                # 如果你怕最后几个窗口太短不想pad，可以加个判断
                # step_samples = max(step_samples, 64)  # 保底最小步长，可选

                m = 0
                chunk_t_points = []
                chunk_psd_points = {ch: [] for ch in target_channels}

                while True:
                    idx_start = m * step_samples
                    idx_end = idx_start + g                     # g == fftpoint

                    if idx_end > L_sig: 
                        break                                 # 最后不足一个窗口直接丢弃（最干净）

                    curr_t = proc_t[idx_start + g//2]          # 用窗口中心时间，更准确

                    for ch in target_channels:
                        x = proc_signals[ch][idx_start:idx_end]
                        x = x - np.mean(x)                     # 去直流
                        P, f = self.psd_me(x, fs_temp, fftpoint)
                        chunk_psd_points[ch].append(P)

                    chunk_t_points.append(curr_t)
                    m += 1
                
                # 收集本块数据
                if len(chunk_t_points) > 0:
                    all_t.extend(chunk_t_points)
                    for ch in target_channels:
                        # 存入 list，稍后合并
                        all_PSD[ch].append(np.array(chunk_psd_points[ch]).T)
                    final_f = f / 1000.0 # 存为 kHz

                print(f"  > Chunk {i+1}/{len(time_chunks)-1} done.")

        # 合并数据
        if not all_t:
            print("No data processed.")
            return None, None, None, None
            
        t_3D = np.array(all_t)
        PSD_list = []
        for ch in target_channels:
            if all_PSD[ch]:
                # hstack 沿着时间轴拼接: (Freq, Time1) + (Freq, Time2) -> (Freq, TotalTime)
                PSD_list.append(np.hstack(all_PSD[ch]))
            else:
                PSD_list.append(np.zeros((len(final_f), len(t_3D))))
                
        return t_3D, final_f, PSD_list, freqs

    # ==================== 静态绘图方法 ====================
    @staticmethod
    def plot_on_axis(ax, t, f, psd, cmap='viridis', title=None):
        """
        将 PSD 数据绘制在指定的 Matplotlib Axes 上
        """
        if psd is None or len(psd) == 0:
            return
        
        # 对数处理，防止 log(0)
        data_log = np.log10(psd + 1e-20)
        
        # 绘图范围 [left, right, bottom, top]
        extent = [t[0], t[-1], f[0], f[-1]]

        vmin = np.percentile(data_log, 5)      # 去掉最暗5%的噪声底
        vmax = np.percentile(data_log, 99.5)    # 去掉最亮0.5%的极端值
        
        im = ax.imshow(data_log, aspect='auto', cmap=cmap, 
                       origin='lower', extent=extent, vmin=vmin, vmax=vmax)
        
        # ax.set_title(title, fontsize=10)
        ax.set_ylabel('Freq (kHz)')
        ax.set_ylim(0, 120)
        return im

# ==============================================================================
#                               使用示例区域
# ==============================================================================

# if __name__ == "__main__":
    
#     # --- 配置 ---
#     shot_num = 83961   # 修改为真实存在的 Shot
#     t_start = 2.8
#     t_end = 3.5         # 时间不宜过长，方便测试
    
#     # 请确保此路径下有数据 (Z盘需要挂载)
#     # 如果在本地测试，请修改 _setup_config 中的路径
    
#     print("=== 开始运行对比示例 ===")

#     # 1. 获取 Judge = 1 (High Res / Original) 的数据
#     analyzer1 = ReflAnalyzer(shot_num, t_start, t_end, judge=1)
#     t1, f1, psds1, freqs1 = analyzer1.run()

#     # 2. 获取 Judge = 2 (Amplitude / Filtered) 的数据
#     analyzer2 = ReflAnalyzer(shot_num, t_start, t_end, judge=2)
#     t2, f2, psds2, freqs2 = analyzer2.run()

#     if t1 is not None and t2 is not None:
#         # 3. 创建画布：4行2列
#         # 左边是 Judge=1，右边是 Judge=2
#         fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True, sharey=True)
        
#         for i in range(4):
#             freq_val = freqs1[i] # 频率值应该是一样的
            
#             # --- 画左边 (Judge=1) ---
#             ax_left = axes[i, 0]
#             title_l = f"#{shot_num} {freq_val}GHz (Original)"
#             ReflAnalyzer.plot_on_axis(ax_left, t1, f1, psds1[i], title_l)
            
#             # --- 画右边 (Judge=2) ---
#             ax_right = axes[i, 1]
#             title_r = f"#{shot_num} {freq_val}GHz (Amp Fluc)"
#             ReflAnalyzer.plot_on_axis(ax_right, t2, f2, psds2[i], title_r)
            
#         # 底部标签
#         axes[-1, 0].set_xlabel('Time (s)')
#         axes[-1, 1].set_xlabel('Time (s)')
        
#         plt.tight_layout()
        
#         # 保存
#         save_file = f"Compare_{shot_num}.png"
#         plt.savefig(save_file, dpi=150)
#         print(f"图像已保存至: {os.path.abspath(save_file)}")
#         plt.show()
        
#     else:
#         print("数据读取失败，请检查文件路径或炮号。")
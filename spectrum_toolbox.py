# spectrum_toolbox.py v2.1 (修复版)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft
from MDSplus import Connection
import os

__version__ = "2.1.0"

class MDSDataLoader:
    def __init__(self, server="202.127.204.12", tree="east"):
        self.server = server
        self.tree = tree
        self._cache = {}
        self._conn = None
    
    def connect(self, shot):
        # 保持兼容：如果外部希望通过 connect 建立上下文连接，
        # 仍然支持但允许传入不同 tree（见下方新增参数）。
        self._conn = Connection(self.server)
        self._conn.openTree(self.tree, shot)
        # 记录当前打开的 tree 以便安全关闭
        self._current_tree = self.tree
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            # 关闭最近打开的 tree（优先使用记录的 _current_tree）
            try:
                close_tree = getattr(self, '_current_tree', self.tree)
                shot = None
                try:
                    shot = self._conn.get('shot')
                except Exception:
                    shot = None
                if shot is not None:
                    self._conn.closeTree(close_tree, shot)
            finally:
                try:
                    self._conn = None
                except Exception:
                    pass
    
    def get_signal(self, shot, signal_path, time_range=None, min_points=100, tree=None):
        """
        从 MDSplus 读取信号，支持按需指定 tree（覆盖实例默认的 tree）。
        参数 `tree` 为 None 时使用实例的 `self.tree`。
        """
        use_tree = tree or self.tree
        cache_key = f"{shot}_{use_tree}_{signal_path}_{time_range}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 为了支持不同 tree，按次打开/关闭 tree，避免长期持有不同 tree 的连接状态
        conn = None
        try:
            conn = Connection(self.server)
            conn.openTree(use_tree, shot)
            data = np.array(conn.get(signal_path))
            time = np.array(conn.get(f'dim_of({signal_path})'))

            if time_range:
                mask = (time >= time_range[0]) & (time <= time_range[1])
                if not mask.any():
                    raise ValueError(f"No data in time range: {time_range}")
                time, data = time[mask], data[mask]

            if len(data) < min_points:
                print(f"⚠️ Attention: {signal_path} only have {len(data)} points")

            self._cache[cache_key] = (time, data)
            return time, data
        except Exception as e:
            raise RuntimeError(f"Read data error {signal_path} (tree={use_tree}): {e}")
        finally:
            if conn:
                try:
                    conn.closeTree(use_tree, shot)
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass


class SpectralAnalyzer:
    @staticmethod
    def calculate_fft_points(sampling_rate, time_window=0.1):
        target_points = int(time_window * sampling_rate)
        N = 2 ** int(np.log2(target_points))
        N = max(64, N)
        return N
    
    @staticmethod
    def compute_psd(data, sampling_rate, time_window=0.1, overlap=0.5):
        n_samples = len(data)
        fs = float(sampling_rate)
        N = SpectralAnalyzer.calculate_fft_points(fs, time_window)
        
        if N > n_samples:
            raise ValueError(f"time_window {time_window}s need {N} points, but only have {n_samples} points")
        
        step = int(N * (1 - overlap))
        step = max(1, step)
        n_segments = max(1, (n_samples - N) // step + 1)
        
        psd_segments = np.zeros((N, n_segments), dtype=complex)
        
        for i in range(n_segments):
            start = i * step
            segment = data[start:start + N]
            segment = segment - np.mean(segment)
            segment *= np.hanning(N)
            psd_segments[:, i] = fft.fft(segment) / N
        
        psd = np.mean(np.abs(psd_segments) ** 2, axis=1)
        freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
        psd = psd[:N//2] * 2
        
        return freqs, psd

    @staticmethod
    def fft_bandpass(data, band_pass_khz, fs):
        """
        对时间序列做频域带通（band_pass_khz 单位为 kHz，格式 [f_low, f_high]）。
        返回与输入同长度的时域信号（可能为复数）。
        """
        band_pass = sorted(band_pass_khz)
        L = len(data)
        # 频谱
        y_data = fft.fft(data)
        # 频率轴 (kHz)
        f = (np.arange(L) * (fs / L) - fs / 2) / 1000.0
        y_data = fft.fftshift(y_data)

        mask = (f < band_pass[0]) | (f > band_pass[1])
        y_data[mask] = 0

        y_data = fft.ifftshift(y_data)
        return fft.ifft(y_data)

    @staticmethod
    def downsample_average(time_arr, data, L_avg):
        """
        将 data 按块平均降采样，返回 (time_avg, data_avg)。
        保持与 NeFluc.refl_average 行为一致。
        """
        if L_avg <= 1:
            return time_arr, data
        num_blocks = len(data) // L_avg
        if num_blocks == 0:
            return time_arr, data
        keep_len = num_blocks * L_avg
        d_reshape = data[:keep_len].reshape((num_blocks, L_avg))
        t_reshape = time_arr[:keep_len].reshape((num_blocks, L_avg))
        return np.mean(t_reshape, axis=1), np.mean(d_reshape, axis=1)
    
    @staticmethod
    def compute_spectrogram(data, sampling_interval, time_window=0.1, overlap=0.5):
        fs = 1.0 / sampling_interval
        n_samples = len(data)
        N = SpectralAnalyzer.calculate_fft_points(fs, time_window)
        
        if N > n_samples:
            raise ValueError(f"time_window {time_window}s Need {N} points, but only have {n_samples} points")
        
        step = int(N * (1 - overlap))
        step = max(1, step)
        n_segments = max(1, (n_samples - N) // step + 1)
        
        psd_matrix = np.zeros((N//2, n_segments))
        time_bins = np.zeros(n_segments)
        
        for i in range(n_segments):
            start = i * step
            segment = data[start:start + N]
            time_bins[i] = (start + N/2) / fs
            freqs, psd = SpectralAnalyzer.compute_psd(segment, fs, time_window, overlap=0)
            psd_matrix[:, i] = 10 * np.log10(psd + 1e-12)
        
        freqs_khz = freqs * 0.001
        spectrogram_df = pd.DataFrame(psd_matrix, index=freqs_khz, columns=time_bins)
        
        return time_bins, freqs_khz, spectrogram_df


class SpectralData:
    """时频数据容器（修复空白bug）"""
    
    def __init__(self, time_bins, freqs, spectrogram_df, shot, signal_path, time_window, time_range):
        self.time = time_bins              # 相对时间 (0-0.6)
        self.freqs = freqs                 # 频率 (kHz)
        self.data = spectrogram_df         # DataFrame (index=频率, columns=相对时间)
        self.shot = shot
        self.signal = signal_path
        self.time_window = time_window
        self.time_range = time_range       # 绝对时间 (2.8-3.4)
    
    def get_absolute_time(self):
        """返回绝对时间"""
        return self.time_range[0] + self.time
    
    def get_spectrum_at_time(self, t):
        """获取指定绝对时间点的频谱"""
        # 转换为相对时间
        rel_t = t - self.time_range[0]
        idx = np.argmin(np.abs(self.time - rel_t))
        return self.data.iloc[:, idx]
    
    def get_frequency_evolution(self, f):
        """获取指定频率的时间演化"""
        # 找到最接近的目标频率
        idx = np.argmin(np.abs(self.freqs - f))
        return self.data.iloc[idx, :]  # 返回该频率在所有时间上的值
    
    def plot(self, ax=None, freq_range=None, time_range=None, cmap='plasma', **kwargs):
        """绘制时频图（修复空白bug）"""
        if ax is None:
            fig, ax = plt.subplots()
        
        # 默认显示完整绝对时间
        plot_data = self.data
        extent_time = self.time_range
        
        # 正确处理time_range子集
        if time_range:
            rel_start = time_range[0] - self.time_range[0]
            rel_end = time_range[1] - self.time_range[0]
            time_mask = (self.time >= rel_start) & (self.time <= rel_end)
            
            if time_mask.any():
                plot_data = plot_data.loc[:, time_mask]
                extent_time = time_range
            else:
                print(f"⚠️ Attention: time_range {time_range} beyond data range {self.time_range}, show full range ")
        
        # 频率切片
        extent_freq = [self.freqs[0], self.freqs[-1]]
        if freq_range:
            plot_data = plot_data[(plot_data.index >= freq_range[0]) & 
                                 (plot_data.index <= freq_range[1])]
            extent_freq = freq_range
        
        # 绘图
        extent = [extent_time[0], extent_time[1], extent_freq[0], extent_freq[-1]]
        im = ax.imshow(plot_data.values, aspect='auto', cmap=cmap,
                      origin='lower', extent=extent, **kwargs)
        
        ax.set_ylabel(f"Freq [kHz]\n{self.signal}")
        ax.set_xlabel("Time [s]")
        ax.set_title(f'#{self.shot} {self.signal} (Δt={self.time_window}s)')
        
        return ax, im


class ProbePlotter:

    SIGNAL_CONFIG = {
        'pxuv': {'prefix': '\\pxuv', 'suffix': '', 'unit': 'V'},
        'khpt': {'prefix': '\\khp', 'suffix': 't', 'unit': 'm/s'},
        'bpol': {'prefix': '\\bpol', 'suffix': '', 'unit': 'T'},
        'ece1-9': {'prefix': '\\hrs0', 'suffix': 'h', 'unit': 'V'},
        'ece10-48': {'prefix': '\\hrs', 'suffix': 'h', 'unit': 'V'},
        'cmplt': {'prefix': '\\cmpl', 'suffix': 't', 'unit': 'V'},
        'cmpt': {'prefix': '\\cmp', 'suffix': 't', 'unit': 'V'},
        'kmpt': {'prefix': '\\kmp', 'suffix': 't', 'unit': 'V'},
        'cmpht': {'prefix': '\\cmph', 'suffix': 't', 'unit': 'V'},
        'kmplt': {'prefix': '\\kmpl', 'suffix': 't', 'unit': 'V'},
        'kmpht': {'prefix': '\\kmph', 'suffix': 't', 'unit': 'V'},
        'cmpln': {'prefix': '\\cmpl', 'suffix': 'n', 'unit': 'V'},
        'cmphn': {'prefix': '\\cmph', 'suffix': 'n', 'unit': 'V'},
        'kmpln': {'prefix': '\\kmpl', 'suffix': 'n', 'unit': 'V'},
        'kmphn': {'prefix': '\\kmph', 'suffix': 'n', 'unit': 'V'},
        'cmput': {'prefix': '\\cmpu', 'suffix': 't', 'unit': 'V'},
        'kmput': {'prefix': '\\kmpu', 'suffix': 't', 'unit': 'V'},
        'cmpun': {'prefix': '\\cmpu', 'suffix': 'n', 'unit': 'V'},
        'kmpun': {'prefix': '\\kmpu', 'suffix': 'n', 'unit': 'V'},
        'cmpdt': {'prefix': '\\cmpd', 'suffix': 't', 'unit': 'V'},
        'kmpdt': {'prefix': '\\kmpd', 'suffix': 't', 'unit': 'V'},
        'cmpdn': {'prefix': '\\cmpd', 'suffix': 'n', 'unit': 'V'},
        'kmpdn': {'prefix': '\\kmpd', 'suffix': 'n', 'unit': 'V'},
        'pointn': {'prefix': '\\point_n', 'suffix': '', 'unit': 'V'},
        'pointf': {'prefix': '\\point_f', 'suffix': '', 'unit': 'V'},
        'dau': {'prefix': '\\dau', 'suffix': '', 'unit': 'V'},
        'dal': {'prefix': '\\dal', 'suffix': '', 'unit': 'V'},
    }
    
    def __init__(self, loader: MDSDataLoader = None):
        self.loader = loader or MDSDataLoader()
        self.analyzer = SpectralAnalyzer()
    
    def compute_spectrogram_data(self, shot, probe_num, signal_name,
                                   time_range, time_window=0.1, overlap=0.5, tree=None,
                                   band_pass=None, average_point=1):
        cfg = self.SIGNAL_CONFIG.get(signal_name)
        if not cfg:
            raise ValueError(f"Unknown Data Type{signal_name}")
        
        signal_path = f"{cfg['prefix']}{probe_num}{cfg['suffix']}"
        time, data = self.loader.get_signal(shot, signal_path, time_range, tree=tree)

        if len(data) == 0:
            raise ValueError(f"{signal_path} No Data")

        ts = np.mean(np.diff(time))

        # 如果提供了带通范围，则先在频域做带通，再取模并可选降采样平均
        data_for_spec = data
        time_for_spec = time
        if band_pass is not None:
            fs = 1.0 / ts
            try:
                filtered = SpectralAnalyzer.fft_bandpass(data, band_pass, fs)
            except Exception as e:
                raise RuntimeError(f"Band_pass failed: {e}")

            # 使用幅度进行时频分析
            magnitude = np.abs(filtered)

            if average_point and int(average_point) > 1:
                time_for_spec, magnitude = SpectralAnalyzer.downsample_average(time, magnitude, int(average_point))
                ts = np.mean(np.diff(time_for_spec)) if len(time_for_spec) > 1 else ts * int(average_point)

            data_for_spec = magnitude

        time_bins, freqs, spec_df = self.analyzer.compute_spectrogram(
            data_for_spec, ts, time_window, overlap
        )
        
        return SpectralData(time_bins, freqs, spec_df, shot, signal_path, time_window, time_range)
    
    def plot_spectrogram(self, ax, shot, probe_num, signal_name,
                        time_range, freq_range, time_window=0.1, 
                        overlap=0.5, tree=None, band_pass=None, average_point=1, **kwargs):
        spec_data = self.compute_spectrogram_data(
            shot, probe_num, signal_name, time_range, time_window, overlap, tree=tree,
            band_pass=band_pass, average_point=average_point
        )
        _, im = spec_data.plot(ax, freq_range=freq_range, **kwargs)
        return im
    
    def plot_spectrogram_grid(self, shot, probe_nums, signal_name,
                               time_range, freq_range, time_window=0.1,
                               overlap=0.5, figsize_per_probe=(8, 1.5),
                               save_path=None, tree=None, band_pass=None, average_point=1):
        cfg = self.SIGNAL_CONFIG.get(signal_name)
        if not cfg:
            raise ValueError(f"Unknown Data Type: {signal_name}")
        
        n_probes = len(probe_nums)
        fig_height = figsize_per_probe[1] * n_probes
        
        fig, axs = plt.subplots(n_probes, 1, 
                               figsize=(figsize_per_probe[0], fig_height), 
                               clear=True, constrained_layout=True)
        
        if n_probes == 1:
            axs = [axs]
        
        for i, probe_num in enumerate(probe_nums):
            ax = axs[i]
            try:
                spec_data = self.compute_spectrogram_data(
                    shot, probe_num, signal_name, time_range, time_window, overlap, tree=tree,
                    band_pass=band_pass, average_point=average_point
                )
                spec_data.plot(ax, freq_range=freq_range)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', ha='center', transform=ax.transAxes)
                continue
            
            ax.set_ylabel(f"Freq [kHz]\n{signal_name}{probe_num}")
            if i == n_probes - 1:
                ax.set_xlabel("Time [s]")
            else:
                ax.set_xticklabels([])
        
        fig.suptitle(f'Shot #{shot} {signal_name.upper()} | Time: {time_range[0]}s-{time_range[1]}s')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Already saved: {save_path}")
        
        return fig, axs


def quick_spectrogram(shot, probe_nums, signal_name, time_range, freq_range,
                     time_window=0.1, overlap=0.5, save_dir=None, tree=None,
                     band_pass=None, average_point=1):
    plotter = ProbePlotter()
    save_path = None
    if save_dir:
        save_path = f"{save_dir}\\{shot}\\{signal_name}_{time_range[0]}-{time_range[1]}s.png"
    return plotter.plot_spectrogram_grid(
        shot, probe_nums, signal_name, time_range, freq_range,
        time_window, overlap, save_path=save_path, tree=tree,
        band_pass=band_pass, average_point=average_point
    )
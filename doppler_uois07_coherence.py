# -*- coding: utf-8 -*-
"""
Doppler judge=2 与 divertor probe UOIS07 的时变相干谱分析
Shot: 80485
Time: 2.9 s - 3.1 s
Freq range: 0 - 25 kHz

说明：
1. Doppler judge=2 不是直接拿 PSD，而是先按照 NeFluc 的逻辑：
   频域带通 -> 取包络(abs) -> 平均降采样。
2. 然后把 Doppler judge=2 包络序列和 UOIS07 原始序列插值到同一个时间网格。
3. 最后对每个滑动时间窗计算 magnitude-squared coherence，得到相关性频谱图。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from NeFluc import ReflAnalyzer
from spectrum_toolbox import MDSDataLoader


# ====================== 科研绘图全局配置 ======================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'SimSun'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.linewidth': 1.0,
})


# ====================== 参数区 ======================
SHOT = 83961
T_START = 2.90
T_END = 3.30

# 若只看某一个 Doppler 通道，可改成 [3]（第4道）
DOPPLER_CHANNELS = [0, 1, 2, 3]

# 仅用于标注；真正频点来自 NeFluc 内部 _get_card_params()
DOPPLER_LABELS_GHZ = [56, 61, 66, 70]

# Doppler judge=2 常用带通（单位 kHz）
DOPPLER_BANDPASS_KHZ = [-3000, -1000]
DOPPLER_AVERAGE_POINT = 4

# 相干谱参数
FREQ_MAX_KHZ = 25.0
MACRO_WIN_SEC = 0.01     # 时间演化窗长，决定时间分辨率与统计稳定性
MACRO_STEP_SEC = 0.005    # 相邻两帧中心时间步长
WELCH_SEG_SEC = 0.005     # 每个宏窗内部 Welch 子段长度
WELCH_OVERLAP = 0.5

SAVE_DIR = r"D:\MyPythonCodes\correlation_plots"
SAVE_NAME = f"shot_{SHOT}_DopplerJ2_UOIS07_coherence_{T_START:.2f}_{T_END:.2f}s.png"


def nextpow2(n: int) -> int:
    n = int(max(1, n))
    return 1 << int(np.ceil(np.log2(n)))


def load_doppler_judge2_timeseries(shot, t_start, t_end, channels,
                                   band_pass_khz=(-3000, -1000), average_point=4):
    """按 NeFluc judge=2 的逻辑，提取 Doppler judge=2 的时域包络序列。"""
    analyzer = ReflAnalyzer(shot, t_start, t_end, judge=2)
    analyzer.card_name = 'Doppler'
    analyzer.band_pass = list(band_pass_khz)
    analyzer.average_point = int(average_point)
    analyzer._update_fs_by_card()

    filename, rows, iq_sign, freqs = analyzer._get_card_params()
    if not filename or not os.path.exists(filename):
        raise FileNotFoundError(f"Doppler bin file not found: {filename}")

    channel_cnt = 8
    read_time = t_start - analyzer.time_delay
    if read_time < 0:
        raise ValueError(f"read_time < 0: {read_time}")

    offset_bytes = int(channel_cnt * analyzer.fs * read_time * 2)
    samples_to_read = int(np.ceil((t_end - t_start) * analyzer.fs))

    with open(filename, 'rb') as fid:
        fid.seek(offset_bytes)
        raw = np.fromfile(fid, dtype=np.int16, count=samples_to_read * channel_cnt)

    if raw.size == 0:
        raise RuntimeError("No Doppler data read from binary file.")

    data_chunk = raw.reshape((channel_cnt, -1), order='F')
    n_samples = data_chunk.shape[1]
    t_raw = t_start + np.arange(n_samples) / analyzer.fs

    results = {}
    for ch in channels:
        r1, r2 = rows[ch]
        sig = data_chunk[r1, :] + iq_sign[ch] * 1j * data_chunk[r2, :]
        t_j2, s_j2 = analyzer._process_amplitude(t_raw, sig)
        results[ch] = {
            'time': np.asarray(t_j2, dtype=float),
            'data': np.asarray(s_j2, dtype=float),
            'label_ghz': freqs[ch],
        }

    return results


def load_uois_signal(shot, probe_num=11, t_start=2.9, t_end=3.3):
    """读取 UOIS07 原始时间序列。"""
    loader = MDSDataLoader(tree='east')
    signal_path = f'\\UIIS{int(probe_num):02d}'
    t, x = loader.get_signal(shot, signal_path, time_range=(t_start, t_end), tree='east')

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]

    if t.size < 10:
        raise RuntimeError(f"Too few UOIS data points: {t.size}")

    order = np.argsort(t)
    return t[order], x[order]


def interpolate_to_common_grid(t1, x1, t2, x2):
    """把两路信号插值到共同均匀时间网格。"""
    dt1 = np.median(np.diff(t1))
    dt2 = np.median(np.diff(t2))
    fs1 = 1.0 / dt1
    fs2 = 1.0 / dt2

    # 共同采样率取两者较低者，避免伪造高频信息
    fs_common = min(fs1, fs2)

    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    n = int(np.floor((t_end - t_start) * fs_common))
    if n < 100:
        raise RuntimeError("Overlap interval too short after interpolation.")

    tg = t_start + np.arange(n) / fs_common
    x1g = np.interp(tg, t1, x1)
    x2g = np.interp(tg, t2, x2)

    return tg, x1g, x2g, fs_common, fs1, fs2


def compute_coherence_spectrogram(x, y, fs, fmax_khz=25.0,
                                  macro_win_sec=0.03,
                                  macro_step_sec=0.005,
                                  welch_seg_sec=0.005,
                                  welch_overlap=0.5):
    """
    计算时变 magnitude-squared coherence。
    返回：times_rel(s), freqs_khz, coh_matrix[freq, time]
    """
    x = signal.detrend(np.asarray(x, dtype=float), type='constant')
    y = signal.detrend(np.asarray(y, dtype=float), type='constant')

    macro_n = int(round(macro_win_sec * fs))
    step_n = int(round(macro_step_sec * fs))
    welch_n = int(round(welch_seg_sec * fs))

    macro_n = max(macro_n, 256)
    step_n = max(step_n, 1)
    welch_n = max(welch_n, 64)

    if welch_n >= macro_n:
        welch_n = max(64, macro_n // 4)

    noverlap = int(round(welch_n * welch_overlap))
    noverlap = min(noverlap, welch_n - 1)
    nfft = nextpow2(welch_n)

    times = []
    coh_list = []
    freqs_khz = None

    for start in range(0, len(x) - macro_n + 1, step_n):
        xs = x[start:start + macro_n]
        ys = y[start:start + macro_n]

        f_hz, cxy = signal.coherence(
            xs, ys,
            fs=fs,
            window='hann',
            nperseg=welch_n,
            noverlap=noverlap,
            nfft=nfft,
            detrend='constant'
        )

        f_khz = f_hz / 1000.0
        mask = (f_khz >= 0) & (f_khz <= fmax_khz)

        if freqs_khz is None:
            freqs_khz = f_khz[mask]

        coh_list.append(cxy[mask])
        times.append((start + macro_n / 2) / fs)

    if len(coh_list) == 0:
        raise RuntimeError("No coherence windows produced. Try enlarging the time range.")

    coh_matrix = np.asarray(coh_list).T
    times = np.asarray(times)

    return times, freqs_khz, coh_matrix


def plot_coherence_maps(results, t_probe, x_probe):
    n_panels = len(results)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.6 * n_panels), sharex=True, constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    ims = []
    meta = []

    for ax, ch in zip(axes, sorted(results.keys())):
        t_dop = results[ch]['time']
        x_dop = results[ch]['data']
        label_ghz = results[ch]['label_ghz']

        tg, xg_dop, xg_probe, fs_common, fs_dop, fs_probe = interpolate_to_common_grid(
            t_dop, x_dop, t_probe, x_probe
        )

        t_rel, f_khz, coh = compute_coherence_spectrogram(
            xg_dop, xg_probe, fs_common,
            fmax_khz=FREQ_MAX_KHZ,
            macro_win_sec=MACRO_WIN_SEC,
            macro_step_sec=MACRO_STEP_SEC,
            welch_seg_sec=WELCH_SEG_SEC,
            welch_overlap=WELCH_OVERLAP
        )

        extent = [tg[0] + t_rel[0], tg[0] + t_rel[-1], f_khz[0], f_khz[-1]]
        im = ax.imshow(
            coh,
            aspect='auto',
            origin='lower',
            extent=extent,
            vmin=0,
            vmax=1,
            cmap='viridis'
        )
        ims.append(im)

        ax.set_ylabel(f'Freq (kHz)\nDoppler {label_ghz} GHz')
        ax.set_ylim(0, FREQ_MAX_KHZ)
        ax.set_xlim(T_START, T_END)
        ax.set_title(f'#{SHOT}  Doppler judge=2 (ch{ch+1}, {label_ghz} GHz)  vs  UOIS07')

        meta.append((ch, fs_dop, fs_probe, fs_common))

    axes[-1].set_xlabel('Time (s)')

    cbar = fig.colorbar(ims[-1], ax=axes, pad=0.01)
    cbar.set_label('Magnitude-squared coherence')

    return fig, axes, meta


if __name__ == '__main__':
    doppler_results = load_doppler_judge2_timeseries(
        SHOT, T_START, T_END,
        channels=DOPPLER_CHANNELS,
        band_pass_khz=DOPPLER_BANDPASS_KHZ,
        average_point=DOPPLER_AVERAGE_POINT,
    )

    t_probe, x_probe = load_uois_signal(
        SHOT, probe_num=7,
        t_start=T_START, t_end=T_END
    )

    fig, axes, meta = plot_coherence_maps(doppler_results, t_probe, x_probe)

    fig.suptitle(
        f'Shot #{SHOT}: Doppler judge=2 vs UOIS07 coherence spectrogram\n'
        f'Time = {T_START:.2f}-{T_END:.2f} s, Freq = 0-{FREQ_MAX_KHZ:.0f} kHz',
        y=1.02
    )

    # 打印采样信息
    print('==== Sampling information ====')
    for ch, fs_dop, fs_probe, fs_common in meta:
        print(
            f'ch{ch+1}: fs_doppler_j2 = {fs_dop/1e3:.2f} kHz, '
            f'fs_uois07 = {fs_probe/1e3:.2f} kHz, '
            f'fs_common = {fs_common/1e3:.2f} kHz'
        )

    nyquist_khz = min(m[3] for m in meta) / 2 / 1e3
    if FREQ_MAX_KHZ >= nyquist_khz:
        print(
            f'WARNING: requested {FREQ_MAX_KHZ:.1f} kHz is at/above common Nyquist '
            f'({nyquist_khz:.2f} kHz). The top edge may be unreliable.'
        )

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved to: {save_path}')
    plt.show()

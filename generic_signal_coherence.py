# -*- coding: utf-8 -*-
"""
通用两路信号相关性频谱图（时变相干谱）脚本

支持两大类数据源：
1. Refl/Reflectometry 类（来自 NeFluc.ReflAnalyzer）
   - Doppler / O_P1 / O_P2 / U_P1 / U_P2 / V_P1 / V_P2 / W_P1 / W_P2 / 5105A/B/C
   - 可选 judge=1 / judge=2
2. MDSplus 类（来自 spectrum_toolbox）
   - bpol / cmplt / cmpt / ece / uois / uiis / liis / lois / pointn / pointf / dau / dal 等

核心思想：
- 把任意两路信号都先整理成统一的时域序列 (t, x)
- 再插值到共同均匀时间轴
- 最后用滑动时间窗计算 magnitude-squared coherence

这样你以后只需要改 SIGNAL_A / SIGNAL_B 两个配置字典即可。
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

from NeFluc import ReflAnalyzer
from spectrum_toolbox import MDSDataLoader, ProbePlotter, SpectralAnalyzer


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


# ====================== 总参数区 ======================
SHOT = 83961
T_START = 3.0
T_END = 3.10
FREQ_MAX_KHZ = 100.0

# 相干谱参数
MACRO_WIN_SEC = 0.005      # 外层滑窗长度，越大越稳，越小时间分辨率越高
MACRO_STEP_SEC = 0.0025    # 相邻两帧中心的时间步长
WELCH_SEG_SEC = 0.005     # 每个宏窗内部 Welch 子段长度
WELCH_OVERLAP = 0.5

# 插值后的共同采样率设置：
# None -> 自动取两路较低采样率
# 数值(Hz) -> 强制使用指定共同采样率
COMMON_FS = None

SAVE_DIR = r"D:\MyPythonCodes\correlation_plots"
SAVE_NAME = f"shot_{SHOT}_generic_signal_coherence_{T_START:.2f}_{T_END:.2f}s.png"


# ====================== 两路信号配置 ======================
# 例1：Doppler judge=2 vs UOIS07
SIGNAL_A = {
    'source': 'refl',
    'label': 'O_P1 judge=2 ch4',
    'card_name': 'O_P1',
    'channel': 4,                    # 人类习惯编号：1~4
    'judge': 2,
    'band_pass_khz': [-1000, -600],
    'average_point': 4,
    'transform': 'real',             # judge=2 输出本来就是实数；这里保留统一接口
}

SIGNAL_B = {
    'source': 'mds',
    'label': 'KHPT06',
    'signal_name': 'khpt',
    'probe_num': 6,
    'tree': 'east',
    'band_pass_khz': None,           # 如需先带通再取包络，可写成 [5, 20] 等（单位 kHz）
    'average_point': 1,
    'transform': 'real',
}

# ---------------------- 其他可直接替换的例子 ----------------------
# 例2：O_P1 judge=1 第1道 vs 磁探针 bpol5
# SIGNAL_A = {
#     'source': 'refl',
#     'label': 'O_P1 judge=1 ch1',
#     'card_name': 'O_P1',
#     'channel': 1,
#     'judge': 1,
#     'band_pass_khz': None,
#     'average_point': 1,
#     'transform': 'real',   # judge=1 是复信号，通常取 real / imag / abs / phase
# }
# SIGNAL_B = {
#     'source': 'mds',
#     'label': 'BPOL05',
#     'signal_name': 'bpol',
#     'probe_num': 5,
#     'tree': 'east',
#     'band_pass_khz': None,
#     'average_point': 1,
#     'transform': 'real',
# }

# 例3：两个 MDS 信号互相关，例如 UOIS07 vs BPOL05
# SIGNAL_A = {
#     'source': 'mds',
#     'label': 'UOIS07',
#     'signal_name': 'uois',
#     'probe_num': 7,
#     'tree': 'east',
#     'band_pass_khz': None,
#     'average_point': 1,
#     'transform': 'real',
# }
# SIGNAL_B = {
#     'source': 'mds',
#     'label': 'BPOL05',
#     'signal_name': 'bpol',
#     'probe_num': 5,
#     'tree': 'east',
#     'band_pass_khz': None,
#     'average_point': 1,
#     'transform': 'real',
# }


@dataclass
class SignalData:
    time: np.ndarray
    data: np.ndarray
    label: str
    fs_est: float
    meta: Dict[str, Any]


def nextpow2(n: int) -> int:
    n = int(max(1, n))
    return 1 << int(np.ceil(np.log2(n)))


def ensure_strictly_increasing_time(t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """排序并移除重复时间点，确保后续插值安全。"""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x)

    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]

    if t.size < 2:
        raise RuntimeError("Too few valid data points after cleaning.")

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 0
    t = t[keep]
    x = x[keep]

    if t.size < 2:
        raise RuntimeError("Time axis is degenerate after removing duplicates.")

    return t, x


def apply_transform(x: np.ndarray, transform: str = 'real') -> np.ndarray:
    """把复/实时域信号变成最终用于相关分析的一维实序列。"""
    transform = (transform or 'real').lower()

    if transform == 'real':
        y = np.real(x)
    elif transform == 'imag':
        y = np.imag(x)
    elif transform == 'abs':
        y = np.abs(x)
    elif transform == 'power':
        y = np.abs(x) ** 2
    elif transform == 'phase':
        y = np.angle(x)
    elif transform == 'unwrap_phase':
        y = np.unwrap(np.angle(x))
    elif transform == 'none':
        # 若是复数，不建议直接送入 coherence；这里默认退化到实部
        y = np.real(x) if np.iscomplexobj(x) else x
    else:
        raise ValueError(f"Unknown transform: {transform}")

    return np.asarray(y, dtype=float)


def read_refl_channel_raw(
    shot: int,
    t_start: float,
    t_end: float,
    card_name: str,
    channel_human: int,
    judge: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    从 ReflAnalyzer 对应的二进制文件中读出指定板卡某一路时域信号。
    channel_human 使用 1~4 编号。
    返回：(t_raw, sig_raw, meta)
    """
    ch = int(channel_human) - 1
    if ch not in [0, 1, 2, 3]:
        raise ValueError("For refl signals, channel must be 1~4.")

    analyzer = ReflAnalyzer(shot, t_start, t_end, judge=judge)
    analyzer.card_name = card_name
    analyzer._update_fs_by_card()

    filename, rows, iq_sign, freqs = analyzer._get_card_params()
    if not filename or not os.path.exists(filename):
        raise FileNotFoundError(f"Reflectometry file not found: {filename}")

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
        raise RuntimeError("No reflectometry data read from binary file.")

    data_chunk = raw.reshape((channel_cnt, -1), order='F')
    n_samples = data_chunk.shape[1]
    t_raw = t_start + np.arange(n_samples) / analyzer.fs

    r1, r2 = rows[ch]
    sig_raw = data_chunk[r1, :] + iq_sign[ch] * 1j * data_chunk[r2, :]

    meta = {
        'source': 'refl',
        'card_name': card_name,
        'channel': channel_human,
        'judge': judge,
        'fs_raw': float(analyzer.fs),
        'freq_label_ghz': freqs[ch],
        'filename': filename,
    }
    return t_raw, sig_raw, meta


def preprocess_refl_signal(
    shot: int,
    t_start: float,
    t_end: float,
    cfg: Dict[str, Any],
) -> SignalData:
    card_name = cfg['card_name']
    channel = cfg['channel']
    judge = int(cfg.get('judge', 1))
    band_pass_khz = cfg.get('band_pass_khz', None)
    average_point = int(cfg.get('average_point', 1))
    transform = cfg.get('transform', 'real')
    label = cfg.get('label', f"{card_name} ch{channel}")

    t_raw, sig_raw, meta = read_refl_channel_raw(
        shot, t_start, t_end, card_name, channel, judge
    )

    analyzer = ReflAnalyzer(shot, t_start, t_end, judge=judge)
    analyzer.card_name = card_name
    analyzer._update_fs_by_card()

    if judge == 2:
        analyzer.band_pass = list(band_pass_khz if band_pass_khz is not None else [-1000, -600])
        analyzer.average_point = average_point
        t_proc, x_proc = analyzer._process_amplitude(t_raw, sig_raw)
        x_proc = apply_transform(x_proc, transform)
    else:
        x_proc = sig_raw
        t_proc = t_raw

        if band_pass_khz is not None:
            x_proc = ReflAnalyzer.fft_bandpass(x_proc, band_pass_khz, analyzer.fs)

        if average_point > 1:
            t_proc, x_proc = ReflAnalyzer.refl_average(t_proc, x_proc, average_point)

        x_proc = apply_transform(x_proc, transform)

    t_proc, x_proc = ensure_strictly_increasing_time(t_proc, x_proc)
    fs_est = 1.0 / np.median(np.diff(t_proc))

    meta.update({
        'band_pass_khz': band_pass_khz,
        'average_point': average_point,
        'transform': transform,
    })
    return SignalData(t_proc, x_proc, label, fs_est, meta)


def build_mds_signal_path(signal_name: Optional[str], probe_num: Optional[int], raw_path: Optional[str]) -> Tuple[str, str]:
    """
    返回 (signal_path, label_suffix)
    - 若给 raw_path，则直接使用
    - 否则走 ProbePlotter.SIGNAL_CONFIG 组装
    """
    if raw_path:
        return raw_path, raw_path

    if signal_name is None:
        raise ValueError("For mds source, either signal_name or raw_path must be provided.")

    plotter = ProbePlotter()
    _, signal_path = plotter._build_signal_path(signal_name, probe_num)
    return signal_path, f"{signal_name}{probe_num}"


def preprocess_mds_signal(
    shot: int,
    t_start: float,
    t_end: float,
    cfg: Dict[str, Any],
) -> SignalData:
    tree = cfg.get('tree', 'east')
    signal_name = cfg.get('signal_name', None)
    probe_num = cfg.get('probe_num', None)
    raw_path = cfg.get('raw_path', None)
    band_pass_khz = cfg.get('band_pass_khz', None)
    average_point = int(cfg.get('average_point', 1))
    transform = cfg.get('transform', 'real')

    signal_path, label_suffix = build_mds_signal_path(signal_name, probe_num, raw_path)
    label = cfg.get('label', label_suffix)

    loader = MDSDataLoader(tree=tree)
    t_raw, x_raw = loader.get_signal(shot, signal_path, time_range=(t_start, t_end), tree=tree)
    t_raw, x_raw = ensure_strictly_increasing_time(t_raw, x_raw)

    x_proc = x_raw
    t_proc = t_raw

    fs_raw = 1.0 / np.median(np.diff(t_raw))

    if band_pass_khz is not None:
        x_proc = SpectralAnalyzer.fft_bandpass(x_proc, band_pass_khz, fs_raw)
        x_proc = np.abs(x_proc)

    if average_point > 1:
        t_proc, x_proc = SpectralAnalyzer.downsample_average(t_proc, x_proc, average_point)

    x_proc = apply_transform(x_proc, transform)
    t_proc, x_proc = ensure_strictly_increasing_time(t_proc, x_proc)
    fs_est = 1.0 / np.median(np.diff(t_proc))

    meta = {
        'source': 'mds',
        'tree': tree,
        'signal_path': signal_path,
        'band_pass_khz': band_pass_khz,
        'average_point': average_point,
        'transform': transform,
        'fs_raw': fs_raw,
    }
    return SignalData(t_proc, x_proc, label, fs_est, meta)


def load_signal_from_config(
    shot: int,
    t_start: float,
    t_end: float,
    cfg: Dict[str, Any],
) -> SignalData:
    source = cfg.get('source', '').lower()
    if source == 'refl':
        return preprocess_refl_signal(shot, t_start, t_end, cfg)
    if source == 'mds':
        return preprocess_mds_signal(shot, t_start, t_end, cfg)
    raise ValueError(f"Unknown source type: {source}")


def interpolate_to_common_grid(
    sig1: SignalData,
    sig2: SignalData,
    common_fs: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """把两路信号插值到共同均匀时间网格。"""
    t1, x1 = sig1.time, sig1.data
    t2, x2 = sig2.time, sig2.data

    fs1 = sig1.fs_est
    fs2 = sig2.fs_est

    if common_fs is None:
        fs_common = min(fs1, fs2)
    else:
        fs_common = float(common_fs)
        if fs_common <= 0:
            raise ValueError("common_fs must be positive.")
        # 不允许超过任一路原始有效采样率，避免伪造高频信息
        fs_common = min(fs_common, fs1, fs2)

    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    if t_end <= t_start:
        raise RuntimeError("The overlap time interval between the two signals is empty.")

    n = int(np.floor((t_end - t_start) * fs_common))
    if n < 100:
        raise RuntimeError("Overlap interval too short after interpolation.")

    tg = t_start + np.arange(n) / fs_common
    x1g = np.interp(tg, t1, x1)
    x2g = np.interp(tg, t2, x2)

    return tg, x1g, x2g, fs_common


def compute_coherence_spectrogram(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    fmax_khz: float = 25.0,
    macro_win_sec: float = 0.02,
    macro_step_sec: float = 0.005,
    welch_seg_sec: float = 0.005,
    welch_overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算时变 magnitude-squared coherence。
    返回：times_rel(s), freqs_khz, coh_matrix[freq, time]
    """
    x = scipy_signal.detrend(np.asarray(x, dtype=float), type='constant')
    y = scipy_signal.detrend(np.asarray(y, dtype=float), type='constant')

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

        f_hz, cxy = scipy_signal.coherence(
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


def plot_coherence_map(
    shot: int,
    sig1: SignalData,
    sig2: SignalData,
    t_grid: np.ndarray,
    t_rel: np.ndarray,
    f_khz: np.ndarray,
    coh: np.ndarray,
    freq_max_khz: float,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)

    extent = [t_grid[0] + t_rel[0], t_grid[0] + t_rel[-1], f_khz[0], f_khz[-1]]
    im = ax.imshow(
        coh,
        aspect='auto',
        origin='lower',
        extent=extent,
        vmin=0,
        vmax=1,
        cmap='viridis'
    )

    ax.set_xlim(T_START, T_END)
    ax.set_ylim(0, freq_max_khz)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Freq (kHz)')
    ax.set_title(f'#{shot}: {sig1.label}  vs  {sig2.label}')

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Magnitude-squared coherence')
    return fig, ax


def main():
    sig1 = load_signal_from_config(SHOT, T_START, T_END, SIGNAL_A)
    sig2 = load_signal_from_config(SHOT, T_START, T_END, SIGNAL_B)

    print('==== Loaded signals ====')
    print(f"A: {sig1.label}")
    print(f"   source = {sig1.meta['source']}")
    print(f"   fs_est = {sig1.fs_est / 1e3:.3f} kHz")
    print(f"   meta   = {sig1.meta}")
    print(f"B: {sig2.label}")
    print(f"   source = {sig2.meta['source']}")
    print(f"   fs_est = {sig2.fs_est / 1e3:.3f} kHz")
    print(f"   meta   = {sig2.meta}")

    t_grid, x1g, x2g, fs_common = interpolate_to_common_grid(sig1, sig2, COMMON_FS)
    print('\n==== Common grid ====')
    print(f'fs_common = {fs_common / 1e3:.3f} kHz')
    print(f't_overlap = [{t_grid[0]:.6f}, {t_grid[-1]:.6f}] s')

    t_rel, f_khz, coh = compute_coherence_spectrogram(
        x1g, x2g,
        fs=fs_common,
        fmax_khz=FREQ_MAX_KHZ,
        macro_win_sec=MACRO_WIN_SEC,
        macro_step_sec=MACRO_STEP_SEC,
        welch_seg_sec=WELCH_SEG_SEC,
        welch_overlap=WELCH_OVERLAP,
    )

    nyquist_khz = fs_common / 2 / 1e3
    if FREQ_MAX_KHZ >= nyquist_khz:
        print(
            f"WARNING: requested {FREQ_MAX_KHZ:.2f} kHz is at/above common Nyquist "
            f"({nyquist_khz:.2f} kHz). The top edge may be unreliable."
        )

    fig, ax = plot_coherence_map(
        SHOT, sig1, sig2,
        t_grid, t_rel, f_khz, coh,
        FREQ_MAX_KHZ,
    )

    fig.suptitle(
        f'Shot #{SHOT}: Generic signal coherence spectrogram\n'
        f'Time = {T_START:.2f}-{T_END:.2f} s, Freq = 0-{FREQ_MAX_KHZ:.1f} kHz',
        y=1.02
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    # save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    # fig.savefig(save_path, dpi=200, bbox_inches='tight')
    # print(f'Saved to: {save_path}')
    plt.show()


if __name__ == '__main__':
    main()

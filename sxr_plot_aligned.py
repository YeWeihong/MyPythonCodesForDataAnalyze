import numpy as np
import matplotlib.pyplot as plt
from MDSplus import Connection
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, filtfilt

# ====================== Params ======================
SHOT = 137569
TIME_START = 8.500
TIME_END = 8.503
SERVER = 'mds.ipp.ac.cn'
TREE_NAME = 'east'
EFIT_TREE = 'efit_east'

# --- 滤波参数 ---
BANDPASS_LOW_FREQ = 1.2e3   # Hz
BANDPASS_HIGH_FREQ = 2.5e3  # Hz
FILTER_ORDER = 4

# --- 绘图参数 ---
FIGSIZE = (10, 15)
RAW_CMAP = 'jet'
FILTERED_CMAP = 'seismic'
RASTERIZED = False

# --- 几何参数 ---
U_Z = np.array([
    0.5132, 0.4933, 0.4732, 0.4527, 0.4319, 0.4107, 0.3893, 0.3675, 0.3453, 0.3228,
    0.3000, 0.2768, 0.2532, 0.2292, 0.2048, 0.1800, 0.1548, 0.1291, 0.1030, 0.0765,
    0.0495, 0.0220, -0.0059, -0.0344, -0.0634, -0.0929, -0.1229, -0.1536, -0.1847,
    -0.2165, -0.2489, -0.2819, -0.3156, -0.3499, -0.3849, -0.4206, -0.4571, -0.4942,
    -0.5322, -0.5709, -0.6105, -0.6509, -0.6922, -0.7344, -0.7775, -0.8216,
], dtype=float)

HRS_CHANNELS = np.arange(1, 47, dtype=int)
U0 = np.array([2.8500, 0.3326], dtype=float)  # [R, Z]
U_R = np.full(U_Z.shape, 1.85, dtype=float)


# ====================== 核心函数 ======================
def bandpass_filter(data, fs, lowcut, highcut, order):
    """应用零相位 Butterworth 带通滤波器。算法保持不变。"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if low >= 1.0 or high >= 1.0 or low <= 0 or high <= 0 or low >= high:
        print(
            f"Warning: normalized frequencies [{low:.4f}, {high:.4f}] invalid "
            f"or out of Nyquist limit. Skip filtering."
        )
        return data

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)


def get_mds_data(conn, node_name, time_window=None):
    """封装 MDSplus 读取，支持时间切片。"""
    try:
        data = np.asarray(conn.get(node_name).data())
        time = np.asarray(conn.get(f'dim_of({node_name})').data())

        if time_window is not None:
            t1, t2 = time_window
            mask = (time >= t1) & (time <= t2)
            return time[mask], data[mask]
        return time, data
    except Exception as e:
        print(f"[WARN] read failed: {node_name} | {e}")
        return None, None


def normalize_signals(data_2d):
    """按通道做 min-max 归一化。算法保持不变。"""
    val_min = np.min(data_2d, axis=1, keepdims=True)
    val_max = np.max(data_2d, axis=1, keepdims=True)
    denom = val_max - val_min
    denom[denom == 0] = 1.0
    return (data_2d - val_min) / denom


def process_geometry_and_mapping(shot, time_center, detector_z, detector_r, pinhole_rz):
    """读取 EFIT 并计算各通道切线最小 rho。算法不变。"""
    cn = None
    try:
        cn = Connection(SERVER)
        cn.openTree(EFIT_TREE, shot)

        r_grid = np.asarray(cn.get(r'\R').data())
        z_grid = np.asarray(cn.get(r'\Z').data())
        psi_rz = np.asarray(cn.get(r'\psirz').data())
        psi_axis = np.asarray(cn.get(r'\ssimag').data())
        psi_bdy = np.asarray(cn.get(r'\ssibry').data())
        t_efit = np.asarray(cn.get(r'dim_of(\ssimag)').data())
    except Exception as e:
        print(f"EFIT data read error: {e}")
        return None, None, None, None
    finally:
        if cn is not None:
            try:
                cn.closeTree(EFIT_TREE, shot)
            except Exception:
                pass

    t_idx = int(np.argmin(np.abs(t_efit - time_center)))
    print(f"Using EFIT at time: {t_efit[t_idx]:.4f} s")

    psi_curr = psi_rz[t_idx, :, :]
    psi_axis_val = psi_axis[t_idx]
    psi_bdy_val = psi_bdy[t_idx]
    psi_norm = (psi_curr - psi_axis_val) / (psi_bdy_val - psi_axis_val)

    interp_func = RegularGridInterpolator(
        (r_grid, z_grid), psi_norm, bounds_error=False, fill_value=None
    )

    rho_tangent = []
    sxru_r_line = np.linspace(1.36, 2.4, 200)

    for z_det, r_det in zip(detector_z, detector_r):
        k = (pinhole_rz[1] - z_det) / (pinhole_rz[0] - r_det)
        sxru_z_line = k * (sxru_r_line - pinhole_rz[0]) + pinhole_rz[1]
        pts = np.column_stack((sxru_r_line, sxru_z_line))

        psi_vals = interp_func(pts)
        valid_psi = psi_vals[~np.isnan(psi_vals)]
        min_psi = np.min(valid_psi) if valid_psi.size > 0 else 1.0
        rho_tangent.append(np.sqrt(np.abs(min_psi)))

    return np.asarray(rho_tangent), psi_norm, r_grid, z_grid


def read_sxr_signals(shot, time_window, channels):
    """
    读取 SXR 通道信号。
    不改变原逻辑：第一个有效通道作为参考时间轴，其他通道必要时插值到该时间轴。
    """
    cn = None
    try:
        cn = Connection(SERVER)
        cn.openTree(TREE_NAME, shot)

        time_vec = None
        signal_list = []
        valid_channels = []

        for ch in channels:
            sig_name = f'\\sxr{ch}u'
            t, sig = get_mds_data(cn, sig_name, time_window)

            if t is None or sig is None or len(t) == 0:
                continue

            if time_vec is None:
                time_vec = t
            elif len(sig) != len(time_vec):
                sig = np.interp(time_vec, t, sig)

            signal_list.append(np.asarray(sig))
            valid_channels.append(ch)

        if time_vec is None or not signal_list:
            raise RuntimeError('No valid SXR channels were loaded.')

        print(f"Loaded {len(valid_channels)}/{len(channels)} channels.")
        return time_vec, np.vstack(signal_list), np.asarray(valid_channels, dtype=int)

    finally:
        if cn is not None:
            try:
                cn.closeTree(TREE_NAME, shot)
            except Exception:
                pass


def centers_to_edges(x):
    """将采样中心点转换为网格边界，用于真实坐标绘图。"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Input centers must be a non-empty 1D array.")
    if x.size == 1:
        dx = 1.0
        return np.array([x[0] - 0.5 * dx, x[0] + 0.5 * dx], dtype=float)

    mids = 0.5 * (x[:-1] + x[1:])
    left = x[0] - 0.5 * (x[1] - x[0])
    right = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.concatenate(([left], mids, [right]))


def setup_axes():
    """使用固定 colorbar 列，确保三个主子图等宽对齐。"""
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.0, 0.045],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.04,
        wspace=0.06,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    cax1 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[1, 1])
    cax3 = fig.add_subplot(gs[2, 1])
    cax3.axis('off')

    return fig, (ax1, ax2, ax3), (cax1, cax2)


def plot_results(time_vec, channel_ids, z_coords, rho_coords, sxr_norm, sxr_filtered):
    """
    优化绘图输出：
    1. 三个子图主绘图区严格等宽对齐；
    2. 横坐标使用真实时间边界，而非仅采样中心；
    3. 明确锁定显示范围到 [TIME_START, TIME_END]。
    """
    fig, axes, caxes = setup_axes()
    ax1, ax2, ax3 = axes
    cax1, cax2 = caxes

    filter_range_khz = f'{BANDPASS_LOW_FREQ / 1e3:.2f}-{BANDPASS_HIGH_FREQ / 1e3:.2f}'
    time_edges = centers_to_edges(time_vec)
    ch_edges = centers_to_edges(channel_ids)
    z_edges = centers_to_edges(z_coords)
    rho_edges = centers_to_edges(rho_coords)

    # Plot 1: Channel vs Time
    im1 = ax1.pcolormesh(
        time_edges,
        ch_edges,
        sxr_norm,
        cmap=RAW_CMAP,
        shading='auto',
        rasterized=RASTERIZED,
    )
    fig.colorbar(im1, cax=cax1, label='Normalized Intensity (Raw)')
    ax1.set_ylabel('Channel ID')
    ax1.set_title(f'SXR Normalized Signals (Shot #{SHOT})')
    ax1.tick_params(labelbottom=False)

    # Plot 2: Z vs Time
    v_max = np.max(np.abs(sxr_filtered)) * 1.05
    im2 = ax2.pcolormesh(
        time_edges,
        z_edges,
        sxr_filtered,
        cmap=FILTERED_CMAP,
        vmin=-v_max,
        vmax=v_max,
        shading='auto',
        rasterized=RASTERIZED,
    )
    fig.colorbar(im2, cax=cax2, label='Normalized Intensity (Filtered)')
    ax2.set_ylabel('Detector Z-coordinate (m)')
    ax2.set_title(f'SXR Filtered Signal (Z vs Time) | Range: {filter_range_khz} kHz')
    ax2.tick_params(labelbottom=False)

    # Plot 3: rho_tangent vs Time
    ax3.pcolormesh(
        time_edges,
        rho_edges,
        sxr_filtered,
        cmap=FILTERED_CMAP,
        vmin=-v_max,
        vmax=v_max,
        shading='auto',
        rasterized=RASTERIZED,
    )
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(r'Normalized Tangential Radius $\rho_{tangent}$')
    ax3.set_title(f'SXR Filtered Signal ($\\rho_{{tangent}}$ vs Time) | Range: {filter_range_khz} kHz')

    # 统一坐标范围：显式锁定真实时间窗
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(TIME_START, TIME_END)
        ax.grid(False)
        ax.margins(x=0)

    ax2.set_ylim(np.min(z_edges), np.max(z_edges))
    ax3.set_ylim(np.min(rho_edges), np.max(rho_edges))

    # 让底部时间轴使用非科学计数法，更贴近实际时间读数
    ax3.ticklabel_format(axis='x', style='plain', useOffset=False)

    plt.show()


# ====================== 主程序 ======================
def main():
    print(f"Processing Shot {SHOT}...")

    t_use = 0.5 * (TIME_START + TIME_END)
    rho_map, _, _, _ = process_geometry_and_mapping(SHOT, t_use, U_Z, U_R, U0)
    if rho_map is None:
        print('Error in EFIT/Geometry processing. Exiting.')
        return

    print('Reading SXR signals...')
    try:
        time_vec, sxr_matrix, valid_channels = read_sxr_signals(
            SHOT, (TIME_START, TIME_END), HRS_CHANNELS
        )
    except Exception as e:
        print(f'SXR Data Read Error: {e}')
        return

    sxr_norm = normalize_signals(sxr_matrix)

    if len(time_vec) > 1:
        dt = float(time_vec[1] - time_vec[0])
        fs = 1.0 / dt
        sxr_filtered = bandpass_filter(
            sxr_norm, fs, BANDPASS_LOW_FREQ, BANDPASS_HIGH_FREQ, FILTER_ORDER
        )
    else:
        sxr_filtered = sxr_norm

    valid_idx = valid_channels - 1
    z_coords = U_Z[valid_idx]
    rho_coords = rho_map[valid_idx]

    print('Plotting...')
    plot_results(
        time_vec=time_vec,
        channel_ids=valid_channels,
        z_coords=z_coords,
        rho_coords=rho_coords,
        sxr_norm=sxr_norm,
        sxr_filtered=sxr_filtered,
    )


if __name__ == '__main__':
    main()

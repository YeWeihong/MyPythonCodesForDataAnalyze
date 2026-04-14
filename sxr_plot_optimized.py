import numpy as np
import matplotlib.pyplot as plt
from MDSplus import Connection
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, filtfilt

# ====================== Params ======================
SHOT = 137569
TIME_START = 10.500
TIME_END = 10.503
SERVER = 'mds.ipp.ac.cn'
TREE_NAME = 'east'
EFIT_TREE = 'efit_east'

# --- 滤波参数 ---
BANDPASS_LOW_FREQ = 1.5e3   # Hz
BANDPASS_HIGH_FREQ = 2.5e3  # Hz
FILTER_ORDER = 4

# --- 绘图参数 ---
FIGSIZE = (10, 15)
CONTOUR_LEVELS = 50
RAW_CMAP = 'jet'
FILTERED_CMAP = 'seismic'

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
    except Exception:
        return None, None


def normalize_signals(data_2d):
    """按通道做 min-max 归一化。算法保持不变。"""
    val_min = np.min(data_2d, axis=1, keepdims=True)
    val_max = np.max(data_2d, axis=1, keepdims=True)
    denom = val_max - val_min
    denom[denom == 0] = 1.0
    return (data_2d - val_min) / denom


def process_geometry_and_mapping(shot, time_center, detector_z, detector_r, pinhole_rz):
    """读取 EFIT 并计算各通道切线最小 rho。算法不变，仅做工程整理。"""
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

        sxr_matrix = np.vstack(signal_list)
        return time_vec, sxr_matrix, np.asarray(valid_channels, dtype=int)

    finally:
        if cn is not None:
            try:
                cn.closeTree(TREE_NAME, shot)
            except Exception:
                pass


def build_grids(time_vec, y_coords):
    """统一生成 contourf 所需二维网格。"""
    time_grid = np.tile(time_vec[None, :], (len(y_coords), 1))
    y_grid = np.tile(y_coords[:, None], (1, len(time_vec)))
    return time_grid, y_grid


def plot_results(time_vec, channel_ids, z_coords, rho_coords, sxr_norm, sxr_filtered):
    """绘制 3 个子图。"""
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)
    ax1, ax2, ax3 = axes

    # Plot 1: Channel vs Time
    t_grid_ch, ch_grid = build_grids(time_vec, channel_ids)
    c1 = ax1.contourf(t_grid_ch, ch_grid, sxr_norm, CONTOUR_LEVELS, cmap=RAW_CMAP)
    fig.colorbar(c1, ax=ax1, label='Normalized Intensity (Raw)')
    ax1.set_ylabel('Channel ID')
    ax1.set_title(f'SXR Normalized Signals (Shot #{SHOT})')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)

    # Plot 2: Z vs Time
    time_grid, z_grid = build_grids(time_vec, z_coords)
    v_max = np.max(np.abs(sxr_filtered)) * 1.05
    c2 = ax2.contourf(
        time_grid, z_grid, sxr_filtered,
        CONTOUR_LEVELS, cmap=FILTERED_CMAP, vmin=-v_max, vmax=v_max
    )
    fig.colorbar(c2, ax=ax2, label='Normalized Intensity (Filtered)')
    filter_range_khz = f'{BANDPASS_LOW_FREQ / 1e3:.2f}-{BANDPASS_HIGH_FREQ / 1e3:.2f}'
    ax2.set_ylabel('Detector Z-coordinate (m)')
    ax2.set_ylim(z_coords.min(), z_coords.max())
    ax2.set_title(f'SXR Filtered Signal (Z vs Time) | Range: {filter_range_khz} kHz')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelbottom=False)

    # Plot 3: rho_tangent vs Time
    _, rho_grid = build_grids(time_vec, rho_coords)
    ax3.contourf(
        time_grid, rho_grid, sxr_filtered,
        CONTOUR_LEVELS, cmap=FILTERED_CMAP, vmin=-v_max, vmax=v_max
    )
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(r'Normalized Tangential Radius $\rho_{tangent}$')
    ax3.set_ylim(rho_coords.min(), rho_coords.max())
    ax3.set_title(f'SXR Filtered Signal ($\\rho_{{tangent}}$ vs Time) | Range: {filter_range_khz} kHz')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
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

    # 使用“实际成功读取的通道”去映射几何量，避免缺道时错位
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

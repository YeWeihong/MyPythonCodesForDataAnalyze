import numpy as np
import matplotlib.pyplot as plt
from MDSplus import Connection
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, filtfilt
import sys

# Params
SHOT = 137569
TIME_START = 10.500
TIME_END = 10.503
SERVER = 'mds.ipp.ac.cn'
TREE_NAME = 'east'
EFIT_TREE = 'efit_east'

# --- 滤波参数 (New Parameters for Bandpass Filter) ---
BANDPASS_LOW_FREQ = 1.5e3   # 低截止频率 (Hz)
BANDPASS_HIGH_FREQ = 2.5e3  # 高截止频率 (Hz)
FILTER_ORDER = 4             # 滤波器阶数 (通常使用 4 或 5 阶)

# --- 几何参数 (Geometry) ---
# SXR Camera Geometry
U_Z = np.array([
    0.5132, 0.4933, 0.4732, 0.4527, 0.4319, 0.4107, 0.3893, 0.3675, 0.3453, 0.3228, 
    0.3, 0.2768, 0.2532, 0.2292, 0.2048, 0.18, 0.1548, 0.1291, 0.103, 0.0765, 
    0.0495, 0.022, -0.0059, -0.0344, -0.0634, -0.0929, -0.1229, -0.1536, -0.1847, 
    -0.2165, -0.2489, -0.2819, -0.3156, -0.3499, -0.3849, -0.4206, -0.4571, -0.4942, 
    -0.5322, -0.5709, -0.6105, -0.6509, -0.6922, -0.7344, -0.7775, -0.8216
])
# 通道选择 (MATLAB 1:46 -> Python 0:46, Python切片含头不含尾)
HRS_CHANNELS = np.arange(1, 47) 
L_CHANNELS = len(HRS_CHANNELS)

U0 = np.array([2.8500, 0.3326]) # Pinhole location [R, Z]
U_R = np.ones(len(U_Z)) * 1.85  # Detector R location

# --- 辅助函数 (保持不变) ---

def bandpass_filter(data, fs, lowcut, highcut, order):
    """
    应用零相位 (filtfilt) Butterworth 带通滤波器。
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if low >= 1.0 or high >= 1.0 or low <= 0 or high <= 0 or low >= high:
        print(f"Warning: Normalized frequencies [{low:.4f}, {high:.4f}] are invalid or out of Nyquist limit (1.0). Skipping filter.")
        return data
        
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)


def get_mds_data(conn, node_name, time_window=None):
    """
     封装 MDSplus读取，包含时间切片功能
    """
    try:
        # 获取信号和时间轴
        data = conn.get(node_name).data()
        time = conn.get(f'dim_of({node_name})').data()
        
        if time_window:
            t1, t2 = time_window
            idx = np.where((time >= t1) & (time <= t2))[0]
            return time[idx], data[idx]
        return time, data
    except Exception as e:
        return None, None

def process_geometry_and_mapping(shot, time_center, U_Z, U_R, U0):
    # --- EFIT 读取和 Psi 归一化 (保持不变) ---
    try:
        cn = Connection(SERVER)
        cn.openTree(EFIT_TREE, shot)
        
        r_grid = cn.get(r'\R').data()
        z_grid = cn.get(r'\Z').data()
        psi_rz = cn.get(r'\psirz').data() 
        psi_axis = cn.get(r'\ssimag').data()
        psi_bdy = cn.get(r'\ssibry').data()
        t_efit = cn.get('dim_of(\ssimag)').data()
        
        cn.closeTree(EFIT_TREE, shot)
    except Exception as e:
        print(f"EFIT data read error: {e}")
        return None, None, None, None

    t_idx = np.argmin(np.abs(t_efit - time_center))
    print(f"Using EFIT at time: {t_efit[t_idx]:.4f} s")

    psi_curr = psi_rz[t_idx, :, :]
    psi_axis_val = psi_axis[t_idx]
    psi_bdy_val = psi_bdy[t_idx]
    psi_norm = (psi_curr - psi_axis_val) / (psi_bdy_val - psi_axis_val)

    interp_func = RegularGridInterpolator((r_grid, z_grid), psi_norm, bounds_error=False, fill_value=None)

    # --- 计算切线最小 Rho (Tangential Rho Calculation) ---
    rho_tangent = []
    sxru_R_line = np.linspace(1.36, 2.4, 200) # 预设 R 范围
    
    for i in range(len(U_Z)):
        # 计算视线 (Line-of-Sight)
        k = (U0[1] - U_Z[i]) / (U0[0] - U_R[i])
        sxru_Z_line = k * (sxru_R_line - U0[0]) + U0[1]
        pts = np.vstack((sxru_R_line, sxru_Z_line)).T
        
        # 沿视线插值 Psi_norm
        psi_vals = interp_func(pts)
        
        # 找到最小 psi_norm 对应的 rho
        valid_psi = psi_vals[~np.isnan(psi_vals)]
        min_psi = np.min(valid_psi) if valid_psi.size > 0 else 1.0
        rho_val = np.sqrt(np.abs(min_psi))
        rho_tangent.append(rho_val)

    return np.array(rho_tangent), psi_norm, r_grid, z_grid

# --- 主程序 ---

def main():
    print(f"Processing Shot {SHOT}...")
    
    # 1. 几何与映射计算
    t_use = (TIME_START + TIME_END) / 2
    # rho_map 包含了每个通道的切线最小归一化半径 rho
    rho_map, psi_map, r_g, z_g = process_geometry_and_mapping(SHOT, t_use, U_Z, U_R, U0)
    
    if rho_map is None:
        print("Error in EFIT/Geometry processing. Exiting.")
        return

    # 2. 读取 SXR 信号 (保持不变)
    print("Reading SXR signals...")
    try:
        cn = Connection(SERVER)
        cn.openTree(TREE_NAME, SHOT)
        
        sxr_data_list = []
        time_vec = None
        
        for ch in HRS_CHANNELS:
            sig_name = f'\\sxr{ch}u'
            t, sig = get_mds_data(cn, sig_name, (TIME_START, TIME_END))
            
            if t is None or len(t) == 0: continue
            
            if time_vec is None: time_vec = t
            
            # 确保时间长度一致
            if len(sig) != len(time_vec):
                sig = np.interp(time_vec, t, sig)
                
            sxr_data_list.append(sig)
            
        cn.closeTree(TREE_NAME, SHOT)
        
        sxr_matrix = np.array(sxr_data_list)
        
    except Exception as e:
        print(f"SXR Data Read Error: {e}")
        return

    # 3. 信号归一化 (Normalization)
    val_min = np.min(sxr_matrix, axis=1, keepdims=True)
    val_max = np.max(sxr_matrix, axis=1, keepdims=True)
    denom = val_max - val_min
    denom[denom == 0] = 1.0
    sxr_norm = (sxr_matrix - val_min) / denom
    
    num_valid_channels = sxr_matrix.shape[0]

    # 4. 滤波处理 (Bandpass Filtering)
    if len(time_vec) > 1:
        dt = time_vec[1] - time_vec[0]
        fs = 1.0 / dt
        sxr_filtered = bandpass_filter(
            sxr_norm, fs, BANDPASS_LOW_FREQ, BANDPASS_HIGH_FREQ, FILTER_ORDER
        )
    else:
        sxr_filtered = sxr_norm 
        
    # --- 5. 绘图 (Visualization) ---
    print("Plotting...")
    fig = plt.figure(figsize=(10, 15)) # 增加图高以容纳 3 个子图
    
    # ----------------------------------------------------
    # Plot 1: SXR Contour (Channel vs Time) - Normalized RAW Data
    # ----------------------------------------------------
    ax1 = fig.add_subplot(3, 1, 1) # 修改为 3 行
    
    channel_ids_valid = HRS_CHANNELS[:num_valid_channels] 
    
    T_grid_ch, CH_grid = np.meshgrid(time_vec, channel_ids_valid)
    
    c1 = ax1.contourf(T_grid_ch, CH_grid, sxr_norm, 50, cmap='jet')
    fig.colorbar(c1, ax=ax1, label='Normalized Intensity (Raw)')
    ax1.set_ylabel('Channel ID')
    ax1.set_title(f'SXR Normalized Signals (Shot #{SHOT})')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False) # 隐藏 x 标签

    # ----------------------------------------------------
    # Plot 2: Z-coordinate vs Time - FILTERED Data
    # ----------------------------------------------------
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1) # 修改为 3 行
    
    Z_coords = U_Z[:num_valid_channels] 
    
    Z_grid = np.tile(Z_coords[:, None], (1, len(time_vec)))
    TIME_grid = np.tile(time_vec[None, :], (num_valid_channels, 1))
    
    DATA_valid = sxr_filtered 
    v_max = np.max(np.abs(DATA_valid)) * 1.05
    
    c2 = ax2.contourf(TIME_grid, Z_grid, DATA_valid, 50, cmap='seismic', vmin=-v_max, vmax=v_max)
    
    filter_range_kHz = f'{BANDPASS_LOW_FREQ/1e3:.2f}-{BANDPASS_HIGH_FREQ/1e3:.2f}'
    fig.colorbar(c2, ax=ax2, label='Normalized Intensity (Filtered)')
    
    ax2.set_ylabel('Detector Z-coordinate (m)') 
    ax2.set_ylim(Z_coords.min(), Z_coords.max()) 
    ax2.set_title(f'SXR Filtered Signal (Z vs Time) | Range: {filter_range_kHz} kHz')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False) # 隐藏 x 标签

    # ----------------------------------------------------
    # Plot 3: Rho_tangent vs Time - FILTERED Data (新增)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1) # 修改为 3 行
    
    Rho_coords = rho_map[:num_valid_channels] 
    
    # 构建 Rho 轴网格 (Channels x Time)
    Rho_grid = np.tile(Rho_coords[:, None], (1, len(time_vec)))
    # TIME_grid 沿用 Plot 2 的
    
    # 使用 Plot 2 相同的滤波数据和颜色范围
    c3 = ax3.contourf(TIME_grid, Rho_grid, DATA_valid, 50, cmap='seismic', vmin=-v_max, vmax=v_max)
    
    # 由于 Plot 2 已经有 Colorbar，Plot 3 可以选择不重复绘制，或使用更紧凑的颜色条
    # fig.colorbar(c3, ax=ax3, label='Normalized Intensity (Filtered)') 
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(r'Normalized Tangential Radius $\rho_{tangent}$') # 使用 LaTeX 渲染 rho 
    ax3.set_ylim(Rho_coords.min(), Rho_coords.max())
    ax3.set_title(r'SXR Filtered Signal ($\rho_{{tangent}}$ vs Time) | Range: {filter_range_kHz} kHz')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    # plt.savefig(f"SXR_Maps_{SHOT}_3.png", dpi=90, bbox_inches='tight') # 如果需要保存，请使用新文件名

if __name__ == '__main__':
    main()
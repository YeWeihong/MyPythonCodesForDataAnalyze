import numpy as np
import matplotlib.pyplot as plt
from MDSplus import Connection
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, filtfilt
import sys

# Params
SHOT = 137569
TIME_START = 8.50
TIME_END = 8.505
SERVER = 'mds.ipp.ac.cn'
TREE_NAME = 'east'
EFIT_TREE = 'efit_east'

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

# --- 辅助函数 ---

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
        # print(f"Error reading {node_name}: {e}")
        return None, None

def process_geometry_and_mapping(shot, time_center, U_Z, U_R, U0):
    """
    读取EFIT平衡数据并计算SXR视线对应的切向半径(Rho)
    """
    try:
        cn = Connection(SERVER)
        cn.openTree(EFIT_TREE, shot)
        
        # 读取网格和平衡量
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

    # 找到最接近的时间点
    t_idx = np.argmin(np.abs(t_efit - time_center))
    print(f"Using EFIT at time: {t_efit[t_idx]:.4f} s")

    # 处理 PSI 数据
    psi_curr = psi_rz[t_idx, :, :]
    
    # 归一化 Psi: (Psi - Psi_axis) / (Psi_bdy - Psi_axis)
    psi_norm = (psi_curr - psi_axis[t_idx]) / (psi_bdy[t_idx] - psi_axis[t_idx])

    # 创建插值器
    interp_func = RegularGridInterpolator((r_grid, z_grid), psi_norm, bounds_error=False, fill_value=None)

    # --- 简化切点 Rho 计算 ---
    rho_tangent = []
    sxru_R_line = np.linspace(1.36, 2.4, 200) 
    
    for i in range(len(U_Z)):
        k = (U0[1] - U_Z[i]) / (U0[0] - U_R[i])
        sxru_Z_line = k * (sxru_R_line - U0[0]) + U0[1]
        pts = np.vstack((sxru_R_line, sxru_Z_line)).T
        psi_vals = interp_func(pts)
        min_psi = np.min(psi_vals) if not np.all(np.isnan(psi_vals)) else 1.0
        rho_val = np.sqrt(np.abs(min_psi))
        rho_tangent.append(rho_val)

    return np.array(rho_tangent), psi_norm, r_grid, z_grid

# --- 主程序 ---

def main():
    print(f"Processing Shot {SHOT}...")
    
    # 1. 几何与映射计算
    t_use = (TIME_START + TIME_END) / 2
    rho_map, psi_map, r_g, z_g = process_geometry_and_mapping(SHOT, t_use, U_Z, U_R, U0)
    
    if rho_map is None:
        return

    # 2. 读取 SXR 信号
    print("Reading SXR signals...")
    try:
        cn = Connection(SERVER)
        cn.openTree(TREE_NAME, SHOT)
        
        sxr_data_list = []
        time_vec = None
        
        for ch in HRS_CHANNELS:
            sig_name = f'\\sxr{ch}u'
            t, sig = get_mds_data(cn, sig_name, (TIME_START, TIME_END))
            
            if t is None or len(t) == 0:
                # print(f"Warning: Channel {ch} no data. Skipping.")
                continue
            
            if time_vec is None:
                time_vec = t
            
            # 简单的重采样或截断以防长度不一致
            if len(sig) != len(time_vec):
                sig = np.interp(time_vec, t, sig)
                
            sxr_data_list.append(sig)
            
        cn.closeTree(TREE_NAME, SHOT)
        
        # 转换为 Numpy 数组 (Channels x Time)
        sxr_matrix = np.array(sxr_data_list)
        
    except Exception as e:
        print(f"SXR Data Read Error: {e}")
        return

    # 3. 信号归一化 (Normalization)
    print("Normalizing data...")
    val_min = np.min(sxr_matrix, axis=1, keepdims=True)
    val_max = np.max(sxr_matrix, axis=1, keepdims=True)
    denom = val_max - val_min
    denom[denom == 0] = 1.0
    sxr_norm = (sxr_matrix - val_min) / denom
    
    # 获取实际读取的通道数量
    num_valid_channels = sxr_matrix.shape[0]

    # 4. 绘图 (Visualization)
    print("Plotting...")
    fig = plt.figure(figsize=(10, 8))
    
    # --- Plot 1: SXR Contour (Channel vs Time) ---
    ax1 = fig.add_subplot(2, 1, 1)
    
    # 仅使用成功读取的通道ID
    channel_ids_valid = HRS_CHANNELS[:num_valid_channels] 
    
    T_grid_ch, CH_grid = np.meshgrid(time_vec, channel_ids_valid)
    
    # 使用 contourf，设置 50 个等级
    c1 = ax1.contourf(T_grid_ch, CH_grid, sxr_norm, 50, cmap='jet')
    fig.colorbar(c1, ax=ax1, label='Normalized Intensity')
    ax1.set_ylabel('Channel ID')
    ax1.set_title(f'SXR raw signals (Shot #{SHOT})')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Re-mapped to Z-coordinate (Z vs Time) ---
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    
    # 获取对应于已读取数据的 Z 坐标
    Z_coords = U_Z[:num_valid_channels] 
    
    # 构建 Z 轴网格 (Channels x Time)
    Z_grid = np.tile(Z_coords[:, None], (1, len(time_vec)))
    
    # 构建 Time 轴网格
    TIME_grid = np.tile(time_vec[None, :], (num_valid_channels, 1))
    
    DATA_valid = sxr_norm
    
    # *** 关键修改: 使用 contourf 替代 pcolormesh ***
    # 使用 contourf，保持 50 个等级与上图一致
    c2 = ax2.contourf(TIME_grid, Z_grid, DATA_valid, 50, cmap='jet')
    
    fig.colorbar(c2, ax=ax2, label='Normalized Intensity')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Detector Z-coordinate (m)') 
    ax2.set_ylim(Z_coords.min(), Z_coords.max()) 
    ax2.set_title(f'SXR Signal (Z vs Time) Shot #{SHOT} (Contourf)')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
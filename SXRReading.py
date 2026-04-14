import numpy as np
import matplotlib.pyplot as plt
from MDSplus import Connection
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.signal import get_window, butter, filtfilt

# ================= 配置参数 =================
SHOT = 73895
TIME_START = 17.0
TIME_END = 19.0
F_CHOOSE_KHZ = 0.97  # 目标频率 (kHz)

# 几何参数
U_Z = np.array([
    0.5132, 0.4933, 0.4732, 0.4527, 0.4319, 0.4107, 0.3893, 0.3675, 0.3453, 0.3228,
    0.3, 0.2768, 0.2532, 0.2292, 0.2048, 0.18, 0.1548, 0.1291, 0.103, 0.0765,
    0.0495, 0.022, -0.0059, -0.0344, -0.0634, -0.0929, -0.1229, -0.1536, -0.1847,
    -0.2165, -0.2489, -0.2819, -0.3156, -0.3499, -0.3849, -0.4206, -0.4571, -0.4942,
    -0.5322, -0.5709, -0.6105, -0.6509, -0.6922, -0.7344, -0.7775, -0.8216
])
D_Z = np.array([
    -0.5077, -0.4879, -0.4679, -0.4476, -0.427, -0.4061, -0.3848, -0.3632, -0.3413,
    -0.319, -0.2964, -0.2734, -0.25, -0.2263, -0.2022, -0.1776, -0.1527, -0.1274,
    -0.1016, -0.0753, -0.0487, -0.0215, 0.0061, 0.0342, 0.0628, 0.092, 0.1216,
    0.1518, 0.1826, 0.2139, 0.2458, 0.2784, 0.3116, 0.3454, 0.3798, 0.415, 0.4509,
    0.4874, 0.5248, 0.5629, 0.6018, 0.6415, 0.6821, 0.7235, 0.7658, 0.8091
])
V_R = np.array([
    1.7299, 1.7464, 1.7631, 1.7798, 1.7965, 1.8133, 1.8302, 1.8471, 1.8641, 1.8811,
    1.8982, 1.9153, 1.9325, 1.9498, 1.9671, 1.9845, 2.002, 2.0195, 2.037, 2.0547,
    2.0724, 2.0901, 2.1079, 2.1258, 2.1437, 2.1617, 2.1798, 2.1979, 2.2161, 2.2344
])

# 针孔位置 (Pinhole locations)
U0 = np.array([2.8500, 0.3326])
D0 = np.array([2.8500, -0.3326])
V0 = np.array([1.8503, 1.4002])

# 探测器固定坐标
U_R = np.ones(len(U_Z)) * 1.85
D_R = np.ones(len(D_Z)) * 1.85
V_Z = np.zeros(len(V_R))

SERVER = 'mds.ipp.ac.cn'

# ================= 辅助函数 =================

def get_mds_data(node_name, tree_name, shot, t_start, t_end):
    """通用 MDSplus 读取函数"""
    try:
        conn = Connection(SERVER)
        conn.openTree(tree_name, shot)
        time = conn.get(f'dim_of({node_name})').data()
        data = conn.get(node_name).data()
        conn.closeTree(tree_name, shot)
        
        idx = np.where((time >= t_start) & (time <= t_end))[0]
        return time[idx], data[idx]
    except Exception as e:
        print(f"Error reading {node_name}: {e}")
        return None, None

def get_efit_data(shot, time_center):
    """读取并处理EFIT平衡数据，返回插值函数"""
    print(f"Reading EFIT for shot {shot} at {time_center}s...")
    try:
        conn = Connection(SERVER)
        conn.openTree('efit_east', shot)
        
        # 读取网格
        r_grid = conn.get(r'\R').data()
        z_grid = conn.get(r'\Z').data()
        # 读取磁通量 (t, R, Z) 或 (t, Z, R)，需检查形状
        psi_rz = conn.get(r'\psirz').data() 
        psi_axis = conn.get(r'\ssimag').data()
        psi_bdy = conn.get(r'\ssibry').data()
        rmaxis = conn.get(r'\rmaxis').data()
        zmaxis = conn.get(r'\zmaxis').data()
        t_efit = conn.get('dim_of(\ssimag)').data()
        conn.closeTree('efit_east', shot)
        
        # 找到最近时刻
        t_idx = np.argmin(np.abs(t_efit - time_center))
        
        # 处理 Psi
        psi_curr = psi_rz[t_idx]
        
        # 转置修正 (确保是 [R, Z] 顺序)
        if psi_curr.shape != (len(r_grid), len(z_grid)):
            psi_curr = psi_curr.T
            
        # 归一化 Psi: (psi - axis) / (bdy - axis) -> 0(轴) ~ 1(边界)
        psi_norm = (psi_curr - psi_axis[t_idx]) / (psi_bdy[t_idx] - psi_axis[t_idx])
        
        # 创建插值器
        interp_psi = RegularGridInterpolator((r_grid, z_grid), psi_norm, bounds_error=False, fill_value=np.nan)
        
        eq_data = {
            'R_grid': r_grid, 'Z_grid': z_grid, 'Psi_norm': psi_norm,
            'R_axis': rmaxis[t_idx], 'Z_axis': zmaxis[t_idx],
            'Interpolator': interp_psi
        }
        return eq_data
    except Exception as e:
        print(f"EFIT Read Error: {e}")
        return None

def find_tangency(Pinhole, Detectors_X, Detectors_Y, eq_data, type='U_or_D'):
    """
    计算视线切点
    Pinhole: [R0, Z0]
    Detectors_X: 探测器R坐标数组
    Detectors_Y: 探测器Z坐标数组
    type: 'U_or_D' (垂直阵列) 或 'V' (水平阵列)
    """
    R_axis = eq_data['R_axis']
    Z_axis = eq_data['Z_axis']
    interp_func = eq_data['Interpolator']
    
    results = {'rho': [], 'theta': [], 'R_tan': [], 'Z_tan': []}
    
    # 定义水平向量用于计算角度 (参考MATLAB逻辑)
    # MATLAB: vec_level=[2.4-R_axis, Z_axis-Z_axis] -> [Pos, 0]
    # 这里直接使用 arctan2 更稳健
    
    num_los = len(Detectors_X)
    
    for i in range(num_los):
        # 1. 构建视线上的采样点
        if type == 'U_or_D':
            # R 从探测器位置到针孔位置延伸到内侧
            # MATLAB: linspace(1.361, 2.4, 1000)
            r_line = np.linspace(1.2, 2.5, 500)
            k = (Pinhole[1] - Detectors_Y[i]) / (Pinhole[0] - Detectors_X[i])
            z_line = k * (r_line - Pinhole[0]) + Pinhole[1]
        else: # type == 'V'
            # Z 从下到上
            z_line = np.linspace(-1.0, 1.2, 500)
            k = (Pinhole[1] - Detectors_Y[i]) / (Pinhole[0] - Detectors_X[i]) # 注意V阵列 X是R，Y是Z
            # line: Z - Z0 = k(R - R0) => R = (Z - Z0)/k + R0
            r_line = (z_line - Pinhole[1]) / k + Pinhole[0]

        # 2. 插值得到 Psi
        points = np.column_stack((r_line, z_line))
        psi_vals = interp_func(points)
        
        # 3. 找最小 Psi (即切点)
        min_idx = np.nanargmin(psi_vals)
        psi_min = psi_vals[min_idx]
        
        # 处理负 Rho 逻辑 (HFS vs LFS)
        # Rho ~ sqrt(psi_norm)
        rho = np.sqrt(np.abs(psi_min))
        
        R_tan = r_line[min_idx]
        Z_tan = z_line[min_idx]
        
        # 判断正负 (Inside/Outside Axis)
        # MATLAB逻辑: if Ru_tang < R_axis: rho = -rho
        if R_tan < R_axis:
            rho = -rho
            
        # 4. 计算极向角 Theta
        # vector from Axis to Tangent Point
        vec_r = R_tan - R_axis
        vec_z = Z_tan - Z_axis
        
        # 使用 arctan2 计算 (-pi, pi)，然后转换到 (0, 2pi) 以匹配习惯
        theta = np.arctan2(vec_z, vec_r)
        if theta < 0:
            theta += 2*np.pi
            
        # 存储结果
        results['rho'].append(rho)
        results['theta'].append(theta)
        results['R_tan'].append(R_tan)
        results['Z_tan'].append(Z_tan)
        
    return results

def get_sxr_matrix(array_name, num_channels, shot, t_start, t_end):
    """
    读取指定阵列的所有通道数据，返回矩阵
    """
    data_matrix = []
    time_axis = None
    valid_channels = []
    
    print(f"Fetching {array_name} array ({num_channels} channels)...")
    
    for i in range(1, num_channels + 1):
        node = f'\\sxr{i}{array_name}' # e.g., \sxr1u
        t, sig = get_mds_data(node, 'east', shot, t_start, t_end)
        
        if t is None or len(t) == 0:
            # 如果读取失败，填充 NaN 或 0，这里选择填充 NaN 行
            # 需要知道时间长度，如果第一个就失败了比较麻烦
            # 暂时先跳过，或者如果 time_axis 已存在则填充
            if time_axis is not None:
                data_matrix.append(np.full(len(time_axis), np.nan))
            else:
                # 如果连第一个都读不到，暂时存个空占位，后面处理
                data_matrix.append(None)
        else:
            if time_axis is None:
                time_axis = t
            
            # 确保时间轴对齐 (简单处理：假设所有通道采样率和长度一致)
            if len(t) != len(time_axis):
                # 长度不一致，进行插值或截断
                # 这里简单截断到最小长度
                min_len = min(len(t), len(time_axis))
                time_axis = time_axis[:min_len]
                # 修正已有的数据
                for k in range(len(data_matrix)):
                    if data_matrix[k] is not None:
                        data_matrix[k] = data_matrix[k][:min_len]
                sig = sig[:min_len]
            
            data_matrix.append(sig)
            valid_channels.append(i)
            
    # 清理 None (如果有通道完全没读到且是开头)
    if time_axis is None:
        return None, None
        
    # 将 None 替换为 NaN 行
    final_matrix = []
    for row in data_matrix:
        if row is None:
            final_matrix.append(np.full(len(time_axis), np.nan))
        else:
            final_matrix.append(row)
            
    return time_axis, np.array(final_matrix)

def perform_svd_analysis(time_axis, data_matrix, n_modes=3):
    """
    对数据矩阵执行 SVD 分解
    X = U * S * V^T
    Topos (Spatial): U * S
    Chronos (Temporal): V
    """
    # 去除 NaN (如果有)
    # 简单的插值填充
    matrix_clean = data_matrix.copy()
    for i in range(matrix_clean.shape[0]):
        row = matrix_clean[i, :]
        nans = np.isnan(row)
        if np.any(nans):
            # 线性插值
            x = np.arange(len(row))
            matrix_clean[i, nans] = np.interp(x[nans], x[~nans], row[~nans])
            
    # 去除平均值 (Perturbed signal)
    means = np.mean(matrix_clean, axis=1, keepdims=True)
    matrix_pert = matrix_clean - means
    
    # SVD
    # matrix_pert shape: (n_channels, n_time)
    # U: (n_channels, n_channels), S: (n_channels,), Vt: (n_channels, n_time)
    # full_matrices=False -> U: (n_channels, K), S: (K,), Vt: (K, n_time) where K=min(M,N)
    U, S, Vt = np.linalg.svd(matrix_pert, full_matrices=False)
    
    topos = U[:, :n_modes] # Spatial structures (columns)
    chronos = Vt[:n_modes, :].T # Temporal evolution (columns)
    singular_values = S[:n_modes]
    
    return topos, chronos, singular_values, matrix_pert

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply Bandpass Filter to 2D data (channels, time)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=1)
    return y

def interpolate_spatially(Z_vals, data_matrix, target_Z):
    """
    Spatially interpolate data to a regular Z grid
    Z_vals: (n_channels,) original Z locations
    data_matrix: (n_channels, n_time)
    target_Z: (n_target,) target Z grid
    """
    # Sort Z_vals for interpolation
    sort_idx = np.argsort(Z_vals)
    Z_sorted = Z_vals[sort_idx]
    data_sorted = data_matrix[sort_idx, :]
    
    # Create interpolator (cubic spline or linear)
    f = interp1d(Z_sorted, data_sorted, kind='cubic', axis=0, fill_value="extrapolate")
    
    return f(target_Z)

# ================= 主程序 =================

def main():
    # 1. 准备磁面和几何
    t_center = (TIME_START + TIME_END) / 2
    eq = get_efit_data(SHOT, t_center)
    
    if eq is None:
        print("Warning: Could not read EFIT data, geometry plots will be skipped.")
        # return # 即使没有 EFIT 也可以画 Channel-Time 图

    # 计算几何 (如果 EFIT 可用)
    geo_U = None
    geo_D = None
    if eq:
        print("Calculating Geometry...")
        geo_U = find_tangency(U0, U_R, U_Z, eq, 'U_or_D')
        geo_D = find_tangency(D0, D_R, D_Z, eq, 'U_or_D')
        # geo_V = find_tangency(V0, V_R, V_Z, eq, 'V')

    # 2. 读取数据矩阵
    # U Array
    t_u, mat_u = get_sxr_matrix('u', len(U_Z), SHOT, TIME_START, TIME_END)
    # D Array
    t_d, mat_d = get_sxr_matrix('d', len(D_Z), SHOT, TIME_START, TIME_END)
    
    # 3. SVD 分析 (Mode Structure)
    # 合并 U 和 D 阵列进行统一分析 (假设时间轴一致)
    if t_u is not None and t_d is not None:
        # 截断到相同长度
        min_len = min(len(t_u), len(t_d))
        t_common = t_u[:min_len]
        mat_u_cut = mat_u[:, :min_len]
        mat_d_cut = mat_d[:, :min_len]
        
        # 拼接: 上半部分 U, 下半部分 D
        mat_combined = np.vstack([mat_u_cut, mat_d_cut])

        # --- Bandpass Filter (0-10kHz) ---
        print("Applying Bandpass Filter (0-10 kHz)...")
        fs = 1 / (t_common[1] - t_common[0])
        # Avoid 0 Hz lowcut if possible, use small value like 10 Hz
        mat_combined = butter_bandpass_filter(mat_combined, 10, 10000, fs, order=4)
        
        print("Performing SVD Analysis...")
        n_modes = 4
        topos, chronos, s_vals, mat_pert = perform_svd_analysis(t_common, mat_combined, n_modes=n_modes)
        
        # 分离 Topos
        topo_u = topos[:len(U_Z), :]
        topo_d = topos[len(U_Z):, :]
        
        # --- Plot SVD Results ---
        fig_svd = plt.figure(figsize=(12, 10))
        fig_svd.suptitle(f'SVD Mode Analysis (Shot #{SHOT})')
        
        # Plot Chronos (Time evolution)
        ax_chronos = plt.subplot(n_modes + 1, 2, (1, 2))
        for k in range(n_modes):
            ax_chronos.plot(t_common, chronos[:, k] * s_vals[k], label=f'Mode {k+1}')
        ax_chronos.set_title('Chronos (Temporal Evolution)')
        ax_chronos.set_xlabel('Time [s]')
        ax_chronos.legend(loc='upper right', fontsize='small')
        ax_chronos.set_xlim(t_common[0], t_common[-1])
        
        # Plot Topos (Spatial Structure)
        # 如果有几何信息，画在 Z_tan 上，否则画在 Channel Index 上
        if eq and geo_U and geo_D:
            Z_U = np.array(geo_U['Z_tan'])
            Z_D = np.array(geo_D['Z_tan'])
            Z_combined = np.concatenate([Z_U, Z_D])
            # 排序以便绘图连线
            sort_idx = np.argsort(Z_combined)
            Z_sorted = Z_combined[sort_idx]
            
            for k in range(n_modes):
                ax_topo = plt.subplot(n_modes + 1, 2, 3 + k)
                # 组合 Topo
                topo_comb = topos[:, k]
                # 绘制
                ax_topo.plot(Z_sorted, topo_comb[sort_idx], 'o-', markersize=3)
                ax_topo.set_title(f'Topo Mode {k+1} (Spatial)')
                ax_topo.set_ylabel('Amplitude')
                if k == n_modes - 1:
                    ax_topo.set_xlabel('Z_tan [m]')
                ax_topo.grid(True)
                
                # 右侧绘制对应的 Chronos Zoom-in (可选) 或 功率谱
                ax_psd = plt.subplot(n_modes + 1, 2, 4 + k)
                f_fft = np.fft.fftfreq(len(t_common), t_common[1]-t_common[0])
                spec = np.fft.fft(chronos[:, k])
                ax_psd.plot(f_fft[:len(f_fft)//2]/1000, np.abs(spec[:len(f_fft)//2]))
                ax_psd.set_title(f'Spectrum Mode {k+1}')
                ax_psd.set_ylabel('PSD')
                ax_psd.set_xlim(0, 50) # 0-50 kHz
                if k == n_modes - 1:
                    ax_psd.set_xlabel('Freq [kHz]')
        else:
            # 无几何信息，画 Channel Index
            pass

        plt.tight_layout()

    # 4. 绘图 (常规)
    
    # --- Plot 1: Channel-Time Contour ---
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    if t_u is not None:
        # 使用 contourf 绘制
        # X: Time, Y: Channel Index
        # extent = [xmin, xmax, ymin, ymax]
        im1 = axs1[0].imshow(mat_u, aspect='auto', origin='lower', 
                             extent=[t_u[0], t_u[-1], 1, len(U_Z)], cmap='jet')
        axs1[0].set_title(f'SXR U Array (Shot #{SHOT})')
        axs1[0].set_ylabel('Channel Index')
        plt.colorbar(im1, ax=axs1[0], label='Intensity')
        
    if t_d is not None:
        im2 = axs1[1].imshow(mat_d, aspect='auto', origin='lower',
                             extent=[t_d[0], t_d[-1], 1, len(D_Z)], cmap='jet')
        axs1[1].set_title(f'SXR D Array (Shot #{SHOT})')
        axs1[1].set_ylabel('Channel Index')
        axs1[1].set_xlabel('Time [s]')
        plt.colorbar(im2, ax=axs1[1], label='Intensity')
        
    plt.tight_layout()
    
    # --- Plot 2: Z-Time Contour (Ripple Map) ---
    if eq and geo_U and geo_D and t_u is not None and t_d is not None:
        print("Generating Z-Time Ripple Map...")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # 1. Prepare Z coordinates
        Z_U = np.array(geo_U['Z_tan'])
        Z_D = np.array(geo_D['Z_tan'])
        Z_all = np.concatenate([Z_U, Z_D])
        
        # 2. Prepare Data (Filtered)
        # mat_combined is already filtered in the SVD section
        # But we need to make sure we are using the same data
        # If SVD section was skipped (e.g. if t_u is None), this might fail.
        # Assuming SVD section ran because we check t_u and t_d.
        
        # 3. Define Regular Grid for Interpolation
        Z_min, Z_max = np.min(Z_all), np.max(Z_all)
        Z_regular = np.linspace(Z_min, Z_max, 200) # 200 spatial points
        
        # 4. Interpolate
        # mat_combined corresponds to Z_all (U then D)
        # Note: mat_combined was filtered.
        ripple_map = interpolate_spatially(Z_all, mat_combined, Z_regular)
        
        # 5. Plot
        # extent = [t_start, t_end, z_min, z_max]
        im2 = ax2.imshow(ripple_map, aspect='auto', origin='lower',
                         extent=[t_common[0], t_common[-1], Z_min, Z_max],
                         cmap='RdBu_r', vmin=-np.std(ripple_map)*3, vmax=np.std(ripple_map)*3)
        
        ax2.set_title(f'SXR Z-Time Ripple Map (0-10kHz) Shot #{SHOT}')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Z_tan [m]')
        plt.colorbar(im2, ax=ax2, label='Perturbed Intensity')
        

        
    # --- Plot 3: Cross-section Geometry ---
    if eq and geo_U and geo_D:
        plt.figure(figsize=(6, 8))
        # 绘制磁面
        plt.contour(eq['R_grid'], eq['Z_grid'], eq['Psi_norm'].T, levels=np.linspace(0, 1, 11), colors='gray', alpha=0.5)
        plt.contour(eq['R_grid'], eq['Z_grid'], eq['Psi_norm'].T, levels=[1.0], colors='k', linewidths=2) # LCFS
        
        # 绘制所有视线
        # U Array
        for i in range(len(U_Z)):
            plt.plot([U0[0], geo_U['R_tan'][i]], [U0[1], geo_U['Z_tan'][i]], 'r-', alpha=0.3, linewidth=0.5)
            plt.plot(geo_U['R_tan'][i], geo_U['Z_tan'][i], 'r.', markersize=2)
            
        # D Array
        for i in range(len(D_Z)):
            plt.plot([D0[0], geo_D['R_tan'][i]], [D0[1], geo_D['Z_tan'][i]], 'b-', alpha=0.3, linewidth=0.5)
            plt.plot(geo_D['R_tan'][i], geo_D['Z_tan'][i], 'b.', markersize=2)
            
        plt.plot(eq['R_axis'], eq['Z_axis'], 'k+', markersize=10, label='Magnetic Axis')
        
        # 标注
        plt.text(U0[0], U0[1], 'U Pinhole', color='r')
        plt.text(D0[0], D0[1], 'D Pinhole', color='b')
        
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.title(f'SXR Chord Geometry (Shot #{SHOT})')
        plt.axis('equal')
        plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
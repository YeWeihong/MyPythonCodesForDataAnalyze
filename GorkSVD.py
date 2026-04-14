import MDSplus as mds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
# %matplotlib inline
# %matplotlib widget   # 如果你喜欢交互式可以打开这行，关掉上面那行

# ========================== 参数 ==========================
server = '202.127.204.12'
shotnum = 137569
time1 = 9.5      # 起始时间 (s)
time2 = 13.0     # 结束时间 (s)
tree = 'east'
R0 = 1.85         # EAST 真实几何中心强烈建议 1.85 m（1.70 会让角度算歪）

# ========================== 读取探针位置 ==========================
magnetic = pd.read_csv('magnetic_probes.csv', index_col=0)

# ========================== MDSplus 连接 & 读取数据 ==========================
conn = mds.Connection(server)
conn.openTree(tree, shotnum)

# 用来保存时间轴（所有探针时间轴一样，取第一个就行）
time_axis = None

# CUT (Upper) 6个
CUT = np.zeros((0,1))
for i in range(1, 7):
    sig_name = f'\\CMPU{i}T'
    data = np.array(conn.get(sig_name))
    t = np.array(conn.get(f'dim_of({sig_name})'))
    idx = np.where((t >= time1) & (t <= time2))[0]
    segment = data[idx].reshape(-1, 1)
    CUT = np.hstack([CUT, segment]) if CUT.shape[1] > 1 else segment
    if time_axis is None:
        time_axis = t[idx]

# CLT (Lower) 10个
CLT = np.zeros((0,1))
for i in range(1, 11):
    sig_name = f'\\CMPL{i}T' if i < 10 else '\\CMPL10T'   # 防止变成 CMPL010T
    data = np.array(conn.get(sig_name))
    segment = data[idx].reshape(-1, 1)   # 直接用上面 CUT 的 idx（时间轴相同）
    CLT = np.hstack([CLT, segment]) if CLT.shape[1] > 1 else segment

# CHT (Horizontal) 8个
CHT = np.zeros((0,1))
for i in range(1, 9):
    sig_name = f'\\CMPH{i}T'
    data = np.array(conn.get(sig_name))
    segment = data[idx].reshape(-1, 1)
    CHT = np.hstack([CHT, segment]) if CHT.shape[1] > 1 else segment

# CDT (Divertor) 6个
CDT = np.zeros((0,1))
for i in range(1, 7):
    sig_name = f'\\CMPD{i}T'
    data = np.array(conn.get(sig_name))
    segment = data[idx].reshape(-1, 1)
    CDT = np.hstack([CDT, segment]) if CDT.shape[1] > 1 else segment

# 合并成 30 × N_time 矩阵（每列一个探针）
CMPs = np.hstack([CUT, CLT, CHT, CDT])   # shape = (N_time, 30)

# ========================== 正确计算每个探针的极向角 theta ==========================
probe_thetas = []
probe_names = []

sections = [('U', 6), ('L', 10), ('H', 8), ('D', 6)]
for prefix, num in sections:
    for i in range(1, num + 1):
        if prefix == 'L' and i == 10:
            probe_name = f'CMPL10T'
        else:
            probe_name = f'CMP{prefix}{i}T'
        probe_names.append(probe_name)
        
        R = magnetic.loc[probe_name, 'R']
        Z = magnetic.loc[probe_name, 'Z']
        theta = np.arctan2(Z, R - R0)          # 自动处理所有象限
        theta = theta % (2 * np.pi)            # 0 ~ 2π
        probe_thetas.append(theta)

probe_thetas = np.array(probe_thetas)   # shape = (30,)

# ========================== SVD 分解 ==========================
U, s, Vh = np.linalg.svd(CMPs, full_matrices=False)   # Vh 是 (30,30)，每行是空间模

# ========================== 奇异值谱 ==========================
plt.figure(figsize=(8,4))
plt.semilogy(range(1, len(s)+1), s, 'o-', lw=2, markersize=8)
plt.xlabel('Mode index')
plt.ylabel('Singular value')
plt.title('SVD Singular Value Spectrum')
plt.grid(True, which="both", ls="--")
plt.show()

# ========================== 时间本征函数（前6个） ==========================
fig, axs = plt.subplots(6, 1, figsize=(10,10), sharex=True)
for i in range(6):
    axs[i].plot(time_axis, U[:, i], 'k', lw=1.2)
    axs[i].set_ylabel(f'Mode {i+1}')
    axs[i].grid(True, alpha=0.4)
axs[-1].set_xlabel('Time (s)')
plt.suptitle('Temporal Modes (U matrix columns)', fontsize=14)
plt.tight_layout()
plt.show()

# ========================== 空间本征函数 vs 极向角（排序后） ==========================
fig, axs = plt.subplots(6, 1, figsize=(10,10), sharex=True)
for i in range(6):
    mode = Vh[i, :]
    sorted_idx = np.argsort(probe_thetas)
    axs[i].plot(probe_thetas[sorted_idx], mode[sorted_idx], 'o-', lw=2)
    axs[i].set_ylabel(f'Mode {i+1}')
    axs[i].grid(True, alpha=0.4)
axs[-1].set_xlabel('Poloidal angle θ (rad)')
axs[-1].set_xticks(np.linspace(0, 2*np.pi, 9))
axs[-1].set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4', '2π'])
plt.suptitle('Spatial Modes vs Poloidal Angle (sorted)', fontsize=14)
plt.tight_layout()
plt.show()

# ========================== 自动计算极向模数 m（最靠谱的傅里叶方法） ==========================
def calc_poloidal_m_spectrum(spatial_mode, thetas, max_m=15):
    N = len(spatial_mode)
    amps = np.zeros(max_m + 1)
    for m in range(max_m + 1):
        if m == 0:
            amps[0] = np.abs(np.sum(spatial_mode)) / N
        else:
            c_m = np.sum(spatial_mode * np.exp(-1j * m * thetas)) / N
            amps[m] = 2 * np.abs(c_m)           # 托卡马克诊断标准幅度
    return amps

print("\n=== 各 SVD 空间模的极向模数谱 ===")
fig, axs = plt.subplots(2, 3, figsize=(15,8))
for mode_idx in range(6):
    spatial_mode = Vh[mode_idx, :]
    spatial_mode = spatial_mode / np.max(np.abs(spatial_mode))   # 归一化便于比较
    
    amps = calc_poloidal_m_spectrum(spatial_mode, probe_thetas, max_m=15)
    dominant_m = np.argmax(amps)
    dominant_amp = amps[dominant_m]
    
    print(f"Mode {mode_idx+1:1d} → 主导极向模数 m = {dominant_m}, 幅度 = {dominant_amp:.4f}")
    
    ax = axs.flat[mode_idx]
    ax.stem(range(len(amps)), amps, use_line_collection=True)
    ax.set_title(f'Mode {mode_idx+1} (dominant m={dominant_m})')
    ax.set_xlabel('m')
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y')

plt.tight_layout()
plt.show()

# ========================== 额外福利：频率分析（对时间模做 FFT） ==========================
fs = 1 / (time_axis[1] - time_axis[0])   # 采样率
freqs = fftfreq(len(time_axis), 1/fs)

fig, axs = plt.subplots(3, 2, figsize=(12,8))
for i in range(6):
    ax = axs.flat[i]
    fft_vals = np.abs(fft(U[:, i]))
    ax.plot(freqs[:len(freqs)//2], fft_vals[:len(len(freqs)//2)])
    ax.set_title(f'Mode {i+1} Frequency Spectrum')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_xlim(0, 50)   # 一般 MHD 频率在 50kHz 以内
    ax.grid(True)
plt.tight_layout()
plt.show()

print("\n运行完毕！你现在直接看打印出来的")
print("Mode 1 → 主导极向模数 m = ?")
print("那一个就是你这个时间窗里最强的极向模数。")
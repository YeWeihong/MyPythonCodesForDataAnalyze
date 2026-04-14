import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
import os

# 导入工具箱
from spectrum_toolbox import MDSDataLoader, ProbePlotter

class ProbeGeometryLoader:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        
    def load_filtered_probes(self, target_prefix):
        """读取Excel并根据规则筛选探针"""
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"找不到文件: {self.excel_path}")
            
        # 支持 csv 或 xlsx
        if self.excel_path.endswith('.csv'):
            df = pd.read_csv(self.excel_path)
        else:
            df = pd.read_excel(self.excel_path)

        df.columns = [c.strip() for c in df.columns]
        
        probe_list = []
        print(f"正在筛选 {target_prefix} 系列探针 (H系列只取第5道)...")
        
        for _, row in df.iterrows():
            # 兼容不同的列名 (name/探针名称, Theta/Angle/Theta/Deg)
            name_col = 'name' if 'name' in df.columns else '探针名称'
            theta_col = 'Theta' if 'Theta' in df.columns else 'Theta/Deg'
            r_col = 'R' if 'R' in df.columns else 'R/m'
            z_col = 'Z' if 'Z' in df.columns else 'Z/m'

            name = str(row[name_col]).strip().upper()
            
            if target_prefix not in name: continue
            if not name.endswith('T'): continue
            
            # H探针过滤规则
            if 'H' in name and '5' not in name:
                continue
            
            mds_path = f"\\{name.lower()}"
            probe_list.append({
                'name': name,
                'path': mds_path,
                'angle': row[theta_col],
                'R': row[r_col],
                'Z': row[z_col]
            })
            
        probe_list.sort(key=lambda x: x['angle'])
        print(f"  -> {target_prefix} 系列共保留 {len(probe_list)} 个探针")
        return probe_list

class PoloidalSVDAnalyzer:
    def __init__(self):
        self.loader = MDSDataLoader()
        self.plotter = ProbePlotter(self.loader)

    def get_signals_matrix(self, shot, probe_list, time_range, tree=None):
        data_matrix = []
        valid_angles, valid_names, valid_R, valid_Z = [], [], [], []
        ref_time = None
        
        for p in probe_list:
            try:
                t, d = self.loader.get_signal(shot, p['path'], time_range, tree=tree)
                d = signal.detrend(d)
                d_norm = d / (np.std(d) + 1e-12)
                
                if ref_time is None:
                    ref_time = t
                    data_matrix.append(d_norm)
                else:
                    if len(d) == 0: continue
                    d_interp = np.interp(ref_time, t, d)
                    d_interp = d_interp / (np.std(d_interp) + 1e-12)
                    data_matrix.append(d_interp)
                
                valid_angles.append(p['angle'])
                valid_names.append(p['name'])
                valid_R.append(p['R'])
                valid_Z.append(p['Z'])
            except:
                continue
        
        if len(data_matrix) > 0:
            return ref_time, np.array(data_matrix), np.array(valid_angles), valid_names, valid_R, valid_Z
        else:
            return None, None, None, None, None, None

    def compute_and_plot(self, shot, prefix, probe_list, time_range, freq_band, save_dir="."):
        """核心计算与绘图逻辑"""
        # 1. 获取数据
        res = self.get_signals_matrix(shot, probe_list, time_range)
        if res[0] is None: return
        t, X, angles, names, R, Z = res
        
        if X.shape[0] < 3:
            print(f"⚠️ {prefix} 系列有效数据太少，跳过")
            return

        # 2. 滤波
        fs = 1 / np.mean(np.diff(t))
        sos = signal.butter(4, freq_band, btype='bandpass', fs=fs, output='sos')
        X_filt = signal.sosfiltfilt(sos, X, axis=1)
        
        # 3. SVD 计算 m
        X_analytic = signal.hilbert(X_filt, axis=1)
        U, s, Vh = np.linalg.svd(X_analytic, full_matrices=False)
        
        spatial_phase = np.angle(U[:, 0])
        unwrapped_phase = np.unwrap(spatial_phase)
        
        theta_rad = np.deg2rad(angles)
        reg = LinearRegression().fit(theta_rad.reshape(-1, 1), unwrapped_phase)
        m_slope = reg.coef_[0]
        intercept = reg.intercept_
        r2 = reg.score(theta_rad.reshape(-1, 1), unwrapped_phase)

        # 4. 绘图
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(f"Poloidal Mode Analysis: {prefix} Series (Shot #{shot})\nFreq: {freq_band}Hz, Time: {time_range}s", fontsize=16, weight='bold')
        gs = fig.add_gridspec(2, 3)

        # 子图1: 几何
        ax1 = fig.add_subplot(gs[0, 0])
        sc = ax1.scatter(R, Z, c=angles, cmap='hsv', s=120, edgecolors='k')
        for i, name in enumerate(names):
            # 简化名称显示
            short_name = name.replace(prefix, '').replace('T', '')
            ax1.text(R[i], Z[i], short_name, fontsize=8)
        ax1.set_title(f"Probe Geometry ({len(names)} probes)")
        ax1.set_xlabel("R [m]")
        ax1.set_ylabel("Z [m]")
        ax1.axis('equal')

        # 子图2: 拟合
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.plot(angles, spatial_phase, 'ro', alpha=0.4, label='Raw Phase')
        ax2.plot(angles, unwrapped_phase, 'bo-', label='Unwrapped Phase')
        fit_line = m_slope * theta_rad + intercept
        ax2.plot(angles, fit_line, 'g--', linewidth=2, label=f'Fit m={m_slope:.2f}')
        ax2.set_title(f"Mode Fit: m = {m_slope:.2f} ($R^2={r2:.2f}$)")
        ax2.set_xlabel("Poloidal Angle [deg]")
        ax2.set_ylabel("Phase [rad]")
        ax2.legend()
        ax2.grid(True)

        # 子图3: 时空图
        ax3 = fig.add_subplot(gs[1, :])
        limit = 2.0
        im = ax3.imshow(X_filt.real, aspect='auto', cmap='RdBu_r',
                        extent=[t[0], t[-1], 0, len(names)], origin='lower',
                        vmin=-limit, vmax=limit)
        
        ax3.set_title(f"Space-Time Plot - Sorted by Angle")
        ax3.set_xlabel("Time [s]")
        ax3.set_yticks(np.arange(len(names)) + 0.5)
        ax3.set_yticklabels(names, fontsize=7)
        
        plt.tight_layout()
        
        # 5. 自动保存
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        filename = f"Poloidal_{prefix}_Shot{shot}_m{m_slope:.2f}.png"
        save_path = os.path.join(save_dir, filename)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 图片已保存: {save_path}")
        
        # 显示并关闭（防止内存泄漏）
        plt.show()
        plt.close(fig)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 请修改为你的实际路径
    EXCEL_PATH = r"D:\MyPythonCodes\magnetic_probes.csv" 
    
    SHOT = 137569
    
    # 建议：将时间范围缩小一点，SVD的效果会更好
    # 例如只看 0.1 秒，能更清楚地看到是否旋转
    TIME_RANGE = [10.0, 11.0] 
    FREQ_BAND = [1500, 2500]

    # 结果保存文件夹
    SAVE_DIR = "SVD_Results"

    geo = ProbeGeometryLoader(EXCEL_PATH)
    analyzer = PoloidalSVDAnalyzer()

    for prefix in ['CMP', 'KMP']:
        print(f"\n=== Processing {prefix} Series ===")
        try:
            probes = geo.load_filtered_probes(prefix)
            
            analyzer.compute_and_plot(
                shot=SHOT, 
                prefix=prefix, 
                probe_list=probes, 
                time_range=TIME_RANGE, 
                freq_band=FREQ_BAND,
                save_dir=SAVE_DIR  # 传入保存路径
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error analyzing {prefix}: {e}")
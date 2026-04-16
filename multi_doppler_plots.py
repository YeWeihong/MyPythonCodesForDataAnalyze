# -*- coding: utf-8 -*-
import os
import csv
import traceback
import numpy as np
import matplotlib.pyplot as plt

from NeFluc import ReflAnalyzer
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


# ======================
# 科研绘图全局配置
# ======================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'axes.unicode_minus': False,

    'font.size': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 14,

    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.linewidth': 1.2
})


# ======================
# 参数区
# ======================
T_START = 2.5
T_END = 3.5
JUDGE = 2
FMAX_KHZ = 125
OUTDIR = "Doppler_plots"

# 你的需求是两个区间：
# 79651-80495 -> 20 MHz
# 80496-97534 -> 10 MHz
# SHOT_RANGES = [
#     (80206, 80495)
# ]
SHOT_RANGES = [
    (80485, 80491) #--- 10 MHz ---
]
# SHOT_RANGES = [
#     (80483, 80486)
# ]

# ======================
# 工具函数
# ======================
def get_fs_by_shot(shot):
    """根据炮号返回采样率"""
    if 79651 <= shot <= 80495:
        return 20 * 10**6
    elif 80496 <= shot <= 97534:
        return 10 * 10**6
    else:
        raise ValueError(f"Shot {shot} 不在指定范围内。")


def build_shot_list():
    """生成完整炮号列表"""
    shots = []
    for s0, s1 in SHOT_RANGES:
        shots.extend(range(s0, s1 + 1))
    return shots


def make_analyzer(shot, t_start, t_end, judge):
    """
    创建 ReflAnalyzer，并强制锁定 Doppler + 指定采样率。
    注意：NeFluc.py 里 run() 会再次调用 _update_fs_by_card()，
    所以这里用 monkey patch 把它锁住，避免 fs 被改回 20 MHz。
    """
    analyzer = ReflAnalyzer(shot, t_start, t_end, judge=judge)

    # 强制 Doppler
    analyzer.card_name = "Doppler"

    # 手动指定采样率
    analyzer.fs = get_fs_by_shot(shot)

    # 锁住，不让 run() 再把 fs 改掉
    analyzer._update_fs_by_card = lambda: None

    # 可选：手动指定参数，保证 judge=1 时分辨率一致
    analyzer.fftpoint_j1 = 2048
    analyzer.step_factor_j1_doppler = 1

    return analyzer


def plot_one_shot(shot, outdir, t_start, t_end, judge, fmax_khz):
    """绘制单炮并保存"""
    analyzer = make_analyzer(shot, t_start, t_end, judge)

    t, f, psd_list, freqs = analyzer.run()

    if t is None or f is None or psd_list is None or freqs is None:
        raise RuntimeError("数据读取失败或未找到对应 bin 文件。")

    if len(psd_list) != 4:
        raise RuntimeError(f"返回的通道数不是 4，而是 {len(psd_list)}。")

    fig, axes = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True)

    for i in range(4):
        ax = axes[i]
        psd = np.asarray(psd_list[i])

        if psd.size == 0:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_ylabel("Freq (kHz)")
            ax.set_ylim(0, fmax_khz)
            continue

        data_log = np.log10(psd + 1e-20)

        vmin = np.percentile(data_log, 5)
        vmax = np.percentile(data_log, 99.5)

        im = ax.imshow(
            data_log,
            aspect='auto',
            origin='lower',
            extent=[t[0], t[-1], f[0], f[-1]],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        ax.set_ylabel("Freq (kHz)")
        ax.set_ylim(0, fmax_khz)
        ax.set_title(
            f"#{shot} Doppler {freqs[i]} GHz (judge={judge}, fs={analyzer.fs/1e6:.0f} MHz)",
            pad=8
        )

        # 时间轴控制
        ax.set_xlim(t_start, t_end)

        # 主刻度每 0.05 s，次刻度每 0.01 s
        # 这样既保留 0.01 s 精度，又不会挤得太乱
        ax.xaxis.set_major_locator(MultipleLocator(0.10))   # 主刻度每 0.10 s
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))   # 次刻度保留 0.01 s 精度
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax.tick_params(axis='x', which='major', labelsize=11, length=6, width=1.0)
        ax.tick_params(axis='x', which='minor', length=3, width=0.8)
        ax.tick_params(axis='y', which='major', labelsize=12, length=6, width=1.0)

        # cbar = fig.colorbar(im, ax=ax, pad=0.01)
        # cbar.set_label("log10(PSD)", fontsize=13)
        # cbar.ax.tick_params(labelsize=12)

    axes[-1].set_xlabel("Time (s)", fontsize=13)
    axes[-1].tick_params(axis='x', labelrotation=30)

    # plt.tight_layout()

    save_name = f"Doppler_{shot}_{t_start:.2f}_{t_end:.2f}s.png"
    save_path = os.path.join(outdir, save_name)
    fig.savefig(save_path, dpi=85)
    plt.close(fig)

    return save_path

def process_one_shot(shot):
    fs_mhz = get_fs_by_shot(shot) / 1e6
    try:
        save_path = plot_one_shot(
            shot=shot,
            outdir=OUTDIR,
            t_start=T_START,
            t_end=T_END,
            judge=JUDGE,
            fmax_khz=FMAX_KHZ
        )
        return (shot, int(fs_mhz), "OK", save_path)
    except Exception as e:
        return (shot, int(fs_mhz), "FAIL", str(e))

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    log_csv = os.path.join(OUTDIR, "batch_log.csv")
    shots = build_shot_list()
    total = len(shots)

    print(f"总任务数: {total}")
    print(f"输出目录: {os.path.abspath(OUTDIR)}")

    max_workers = 3   # 先从 2 开始，不要一上来开 4

    with open(log_csv, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["shot", "fs_MHz", "status", "message"])

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one_shot, shot): shot for shot in shots}

            done_count = 0
            for future in as_completed(futures):
                result = future.result()
                shot, fs_mhz, status, message = result

                done_count += 1
                print(f"[{done_count}/{total}] shot={shot} | {status}")

                writer.writerow([shot, fs_mhz, status, message])
                fcsv.flush()

    print("批处理完成。")


if __name__ == "__main__":
    mp.freeze_support()
    main()
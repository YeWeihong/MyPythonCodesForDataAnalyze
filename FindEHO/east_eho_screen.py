# -*- coding: utf-8 -*-
"""
EAST EHO candidate coarse screener
----------------------------------
用途：
1. 批量读取 EAST 炮号的基础慢信号
2. 自动寻找 H-mode 候选时间窗
3. 根据 RF 主导、q95、Dalpha 安静程度、密度等指标评分
4. 输出 candidate_summary.csv
5. 为每炮保存 overview 图，便于人工复核

依赖：
    pip install numpy pandas matplotlib MDSplus

示例：
    python east_eho_screen.py --start 80400 --end 80550 --outdir ./screen_out

或者：
    python east_eho_screen.py --shot-file shots.txt --outdir ./screen_out
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from MDSplus import Connection
except ImportError as e:
    raise ImportError(
        "未检测到 MDSplus Python 包。请先安装：pip install MDSplus"
    ) from e


# =========================
# 全局绘图配置
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimSun"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.0,
})


# =========================
# 可修改配置
# =========================
MDS_HOST = "202.127.204.12"


@dataclass
class SignalSpec:
    tree: str
    node: str
    scale: float = 1.0
    time_expr: Optional[str] = None
    required: bool = True   # 如果 dim_of(node) 不适用，可单独指定时间表达式


@dataclass
class ScreenConfig:
    t_min: float = 0.0
    t_max: float = 10.0
    dt: float = 1e-3  # 1 ms

    # 统计/研究时间窗策略
    # full: 全程
    # fixed: 固定时间窗 [analysis_t1, analysis_t2]
    # ip_stable: 根据 Ip 平顶区自动选窗
    analysis_mode: str = "full"
    analysis_t1: Optional[float] = None
    analysis_t2: Optional[float] = None
    ip_stable_frac: float = 0.90      # 平顶判据: |Ip| >= frac * max(|Ip|)
    ip_smooth_window: float = 0.05    # Ip 平滑窗口(s)

    # H-mode 粗判据
    baseline_window: float = 0.10      # 100 ms
    dalpha_drop_ratio: float = 0.80
    wmhd_rise_ratio: float = 1.10
    min_hmode_duration: float = 0.05   # 50 ms

    # q95 优先窗口
    q95_low: float = 3.5
    q95_high: float = 5.5

    # RF 主导阈值
    rf_ratio_threshold: float = 0.70

    # Dalpha 安静程度阈值
    quiet_index_threshold: float = 0.70

    # 密度下限（这里只是初始经验值，后续可校准）
    ne_min: float = 2.0e19

    # overview 作图时间范围；None 表示全程
    plot_t_min: Optional[float] = None
    plot_t_max: Optional[float] = None

    # 是否只给 A/B 类画图
    plot_only_candidates: bool = False


# =========================
# 节点映射
# =========================
# 你最需要改的是 Dalpha 和 q95 这两个节点
SIGNAL_MAP: Dict[str, SignalSpec] = {
    "Ip": SignalSpec("pcs_east", r"\pcrl01", required=True),
    "ne": SignalSpec("pcs_east", r"\DFSDEV", required=True),
    "WMHD": SignalSpec("energy_east", r"\eng", required=True),
    "q95": SignalSpec("efitrt_east", r"\q95", required=True),
    "Dalpha": SignalSpec("east", r"\Dau2", required=True),

    "PLHI1": SignalSpec("east_1", r"\PLHI1", required=False),
    "PLHR1": SignalSpec("east_1", r"\PLHR1", required=False),
    "PLHI2": SignalSpec("east_1", r"\PLHI2", required=False),
    "PLHR2": SignalSpec("east_1", r"\PLHR2", required=False),

    "ECRH1": SignalSpec("ECRH_EAST", r"\ECRH1", required=False),
    "ECRH2": SignalSpec("ECRH_EAST", r"\ECRH2", required=False),
    "ECRH3": SignalSpec("ECRH_EAST", r"\ECRH3", required=False),

    "PNBI1L": SignalSpec("nbi_east", r"\PNBI1LSOURCE", required=False),
    "PNBI1R": SignalSpec("nbi_east", r"\PNBI1RSOURCE", required=False),
    "PNBI2L": SignalSpec("nbi_east", r"\PNBI2LSOURCE", required=False),
    "PNBI2R": SignalSpec("nbi_east", r"\PNBI2RSOURCE", required=False),
}

# =========================
# MDSplus 读取
# =========================
class EASTMDSReader:
    def __init__(self, host: str = MDS_HOST):
        self.host = host
        self.conn = Connection(host)

    def read_signal(self, shot: int, spec: SignalSpec) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 MDSplus 读取一个信号及其时间轴。
        返回:
            t: 1D np.ndarray
            y: 1D np.ndarray
        """
        try:
            self.conn.openTree(spec.tree, shot)
            y = np.asarray(self.conn.get(spec.node).data(), dtype=float).ravel()

            if spec.time_expr is not None:
                t = np.asarray(self.conn.get(spec.time_expr).data(), dtype=float).ravel()
            else:
                t = np.asarray(self.conn.get(f"dim_of({spec.node})").data(), dtype=float).ravel()

            self.conn.closeTree(spec.tree, shot)

            if y.size == 0 or t.size == 0:
                raise ValueError(f"空信号: tree={spec.tree}, node={spec.node}")

            # 有些信号数据与时间长度可能不完全一致，这里尽量截成一致长度
            n = min(len(t), len(y))
            t = t[:n]
            y = y[:n] * spec.scale
            return t, y

        except Exception as e:
            try:
                self.conn.closeTree(spec.tree, shot)
            except Exception:
                pass
            raise RuntimeError(
                f"读取失败: shot={shot}, tree={spec.tree}, node={spec.node}, err={e}"
            ) from e

    def read_all_signals(self, shot: int, signal_map: Dict[str, SignalSpec]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for name, spec in signal_map.items():
            try:
                out[name] = self.read_signal(shot, spec)
            except Exception as e:
                if spec.required:
                    raise RuntimeError(
                        f"必需信号读取失败: name={name}, tree={spec.tree}, node={spec.node}, err={e}"
                    ) from e
                else:
                    logging.warning(
                        "可选信号缺失，按 0 处理: shot=%d, name=%s, tree=%s, node=%s",
                        shot, name, spec.tree, spec.node
                    )
                    # 返回一个最简单的零信号，占位即可
                    out[name] = (np.array([0.0, 10.0], dtype=float),
                                np.array([0.0, 0.0], dtype=float))

        return out


# =========================
# 工具函数
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resample_to_grid(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    将信号插值到统一时间网格。
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(t_grid, np.nan, dtype=float)

    t_valid = t[mask]
    y_valid = y[mask]

    # 排序 + 去重，避免 interp 出错
    order = np.argsort(t_valid)
    t_valid = t_valid[order]
    y_valid = y_valid[order]

    unique_t, unique_idx = np.unique(t_valid, return_index=True)
    unique_y = y_valid[unique_idx]

    if unique_t.size < 2:
        return np.full_like(t_grid, np.nan, dtype=float)

    y_grid = np.interp(t_grid, unique_t, unique_y, left=np.nan, right=np.nan)
    return y_grid


def contiguous_regions(mask: np.ndarray, t: np.ndarray) -> List[Tuple[float, float, int, int]]:
    """
    将布尔 mask 转换为连续时间区间列表：
    [(t1, t2, i1, i2), ...]
    """
    regions: List[Tuple[float, float, int, int]] = []
    if len(mask) == 0:
        return regions

    start_idx = None
    for i, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = i
        elif (not flag) and start_idx is not None:
            regions.append((t[start_idx], t[i - 1], start_idx, i - 1))
            start_idx = None

    if start_idx is not None:
        regions.append((t[start_idx], t[-1], start_idx, len(t) - 1))

    return regions


def rolling_median(x: np.ndarray, window_pts: int) -> np.ndarray:
    """
    使用 pandas 实现 rolling median，并做前后填充。
    """
    s = pd.Series(x)
    out = s.rolling(window_pts, min_periods=max(3, window_pts // 5)).median()
    out = out.bfill().ffill()
    return out.to_numpy(dtype=float)


def build_heating_signals(sig: Dict[str, np.ndarray]) -> None:
    """
    在 sig 原地新增:
        P_LH, P_ECRH, P_NBI, P_RF, P_total
    """
    def safe(arr_name: str) -> np.ndarray:
        arr = sig.get(arr_name)
        if arr is None:
            raise KeyError(f"缺少信号: {arr_name}")
        return np.nan_to_num(arr, nan=0.0)

    sig["P_LH"] = safe("PLHI1") + safe("PLHR1") + safe("PLHI2") + safe("PLHR2")
    sig["P_ECRH"] = safe("ECRH1") + safe("ECRH2") + safe("ECRH3")
    sig["P_NBI"] = safe("PNBI1L") + safe("PNBI1R") + safe("PNBI2L") + safe("PNBI2R")
    sig["P_RF"] = sig["P_LH"] + sig["P_ECRH"]
    sig["P_total"] = sig["P_RF"] + sig["P_NBI"]


def build_analysis_mask(
    t: np.ndarray,
    ip: np.ndarray,
    cfg: ScreenConfig
) -> Tuple[np.ndarray, float, float, str]:
    """
    构建分析窗口掩码（用于限制“只研究某时间段”）。
    返回:
        mask, t1, t2, mode_used
    """
    full_mask = np.ones_like(t, dtype=bool)
    full_t1, full_t2 = float(t[0]), float(t[-1])
    mode = (cfg.analysis_mode or "full").lower()

    if mode == "full":
        return full_mask, full_t1, full_t2, "full"

    if mode == "fixed":
        if cfg.analysis_t1 is None or cfg.analysis_t2 is None:
            raise ValueError("analysis_mode=fixed 时必须设置 analysis_t1 与 analysis_t2")
        t1 = max(full_t1, min(cfg.analysis_t1, cfg.analysis_t2))
        t2 = min(full_t2, max(cfg.analysis_t1, cfg.analysis_t2))
        if t2 <= t1:
            raise ValueError(f"固定分析窗口无效: [{t1}, {t2}]")
        mask = (t >= t1) & (t <= t2)
        return mask, float(t1), float(t2), "fixed"

    if mode == "ip_stable":
        ip_abs = np.abs(np.nan_to_num(ip, nan=0.0))
        n_win = max(5, int(cfg.ip_smooth_window / cfg.dt))
        ip_smooth = rolling_median(ip_abs, n_win)

        if not np.isfinite(ip_smooth).any():
            return full_mask, full_t1, full_t2, "full_fallback"

        peak = float(np.nanmax(ip_smooth))
        if (not np.isfinite(peak)) or peak <= 0:
            return full_mask, full_t1, full_t2, "full_fallback"

        thr = cfg.ip_stable_frac * peak
        stable_mask = ip_smooth >= thr
        regions = contiguous_regions(stable_mask, t)

        if len(regions) == 0:
            return full_mask, full_t1, full_t2, "full_fallback"

        t1, t2, _, _ = max(regions, key=lambda r: (r[1] - r[0]))
        mask = (t >= t1) & (t <= t2)
        return mask, float(t1), float(t2), "ip_stable"

    raise ValueError(f"未知 analysis_mode: {cfg.analysis_mode}")


def detect_hmode_windows(
    t: np.ndarray,
    dalpha: np.ndarray,
    wmhd: np.ndarray,
    ne: np.ndarray,
    cfg: ScreenConfig,
    analysis_mask: Optional[np.ndarray] = None
) -> List[Tuple[float, float, int, int]]:
    """
    基于“相对前一段基线”的 Dalpha 下降 + WMHD 上升，
    寻找 H-mode 候选时间窗。
    """
    n_base = max(5, int(cfg.baseline_window / cfg.dt))

    d_smooth = pd.Series(dalpha).rolling(n_base, min_periods=max(3, n_base // 5)).median().bfill().ffill().to_numpy()
    w_smooth = pd.Series(wmhd).rolling(n_base, min_periods=max(3, n_base // 5)).median().bfill().ffill().to_numpy()

    d_base = np.roll(d_smooth, n_base)
    w_base = np.roll(w_smooth, n_base)
    d_base[:n_base] = np.nan
    w_base[:n_base] = np.nan

    cond1 = np.isfinite(d_smooth) & np.isfinite(d_base) & (d_smooth < cfg.dalpha_drop_ratio * d_base)
    cond2 = np.isfinite(w_smooth) & np.isfinite(w_base) & (w_smooth > cfg.wmhd_rise_ratio * w_base)
    cond3 = np.isfinite(ne)

    mask = cond1 & cond2 & cond3
    if analysis_mask is not None:
        if analysis_mask.shape != mask.shape:
            raise ValueError("analysis_mask 与时间轴长度不一致")
        mask = mask & analysis_mask

    regions = contiguous_regions(mask, t)
    valid_regions = []
    for t1, t2, i1, i2 in regions:
        if (t2 - t1) >= cfg.min_hmode_duration:
            valid_regions.append((t1, t2, i1, i2))

    return valid_regions

def extract_main_window_features(
    t: np.ndarray,
    sig: Dict[str, np.ndarray],
    hwin: Tuple[float, float, int, int],
    cfg: ScreenConfig,
    analysis_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    针对一个 H-mode 候选时间窗提取特征。
    q95_mean 使用 analysis_mask（若给定），否则用 hwin。
    """
    t1, t2, i1, i2 = hwin
    mask = np.zeros_like(t, dtype=bool)
    mask[i1:i2 + 1] = True

    stat_mask = mask
    if analysis_mask is not None and analysis_mask.shape == mask.shape and np.any(analysis_mask):
        stat_mask = analysis_mask

    pre_t1 = max(cfg.t_min, t1 - cfg.baseline_window)
    pre_mask = (t >= pre_t1) & (t < t1)

    q95_mean = float(np.nanmean(sig["q95"][stat_mask])) if np.any(stat_mask) else np.nan
    ne_mean = float(np.nanmean(sig["ne"][mask])) if np.any(mask) else np.nan

    p_rf = float(np.nanmean(sig["P_RF"][mask])) if np.any(mask) else np.nan
    p_total = float(np.nanmean(sig["P_total"][mask])) if np.any(mask) else np.nan
    rf_ratio = p_rf / (p_total + 1e-12)

    d_std_h = float(np.nanstd(sig["Dalpha"][mask])) if np.any(mask) else np.nan
    d_std_pre = float(np.nanstd(sig["Dalpha"][pre_mask])) if np.any(pre_mask) else np.nan
    quiet_index = d_std_h / (d_std_pre + 1e-12)

    return {
        "hmode_t1": t1,
        "hmode_t2": t2,
        "hmode_duration": t2 - t1,
        "q95_mean": q95_mean,
        "ne_mean": ne_mean,
        "rf_ratio": rf_ratio,
        "quiet_index": quiet_index,
    }

# ...existing code...

def score_features(feat: Dict[str, float], cfg: ScreenConfig) -> Tuple[int, str]:
    """
    根据特征评分并打标签。
    A: 强候选
    B: 边缘候选
    C: 低优先级
    """
    score = 0

    if np.isfinite(feat["hmode_duration"]) and feat["hmode_duration"] >= 0.05:
        score += 3

    if np.isfinite(feat["rf_ratio"]) and feat["rf_ratio"] >= cfg.rf_ratio_threshold:
        score += 2

    if np.isfinite(feat["q95_mean"]) and cfg.q95_low <= feat["q95_mean"] <= cfg.q95_high:
        score += 2

    if np.isfinite(feat["quiet_index"]) and feat["quiet_index"] <= cfg.quiet_index_threshold:
        score += 2

    if np.isfinite(feat["ne_mean"]) and feat["ne_mean"] >= cfg.ne_min:
        score += 1

    if np.isfinite(feat["hmode_duration"]) and feat["hmode_duration"] >= 0.10:
        score += 1

    if score >= 8:
        label = "A"
    elif score >= 5:
        label = "B"
    else:
        label = "C"

    return score, label

# ...existing code...

def choose_best_window(
    t: np.ndarray,
    sig: Dict[str, np.ndarray],
    wins: List[Tuple[float, float, int, int]],
    cfg: ScreenConfig,
    analysis_mask: Optional[np.ndarray] = None
) -> Tuple[Tuple[float, float, int, int], Dict[str, float], int, str]:
    """
    从多个候选 H-mode 时间窗中选择得分最高的那个。
    """
    best_win = None
    best_feat = None
    best_score = -1
    best_label = "C"

    for win in wins:
        feat = extract_main_window_features(t, sig, win, cfg, analysis_mask=analysis_mask)
        score, label = score_features(feat, cfg)
        if score > best_score:
            best_score = score
            best_label = label
            best_feat = feat
            best_win = win

    if best_win is None or best_feat is None:
        raise RuntimeError("未能从候选时间窗中选出最佳窗口")

    return best_win, best_feat, best_score, best_label


def save_overview_plot(
    shot: int,
    t: np.ndarray,
    sig: Dict[str, np.ndarray],
    feat: Optional[Dict[str, float]],
    score: int,
    label: str,
    outpath: Path,
    cfg: ScreenConfig
) -> None:
    """
    保存 overview 图。
    """
    if cfg.plot_t_min is None:
        plot_t_min = t[0]
    else:
        plot_t_min = cfg.plot_t_min

    if cfg.plot_t_max is None:
        plot_t_max = t[-1]
    else:
        plot_t_max = cfg.plot_t_max

    mask = (t >= plot_t_min) & (t <= plot_t_max)
    tt = t[mask]

    fig, axes = plt.subplots(
        nrows=6, ncols=1, figsize=(8, 8), sharex=True, constrained_layout=True
    )

    axes[0].plot(tt, sig["Ip"][mask], linewidth=1.0)
    axes[0].set_ylabel("Ip")

    axes[1].plot(tt, sig["q95"][mask], linewidth=1.0)
    axes[1].set_ylabel("q95")

    axes[2].plot(tt, sig["ne"][mask], linewidth=1.0)
    axes[2].set_ylabel("ne")

    axes[3].plot(tt, sig["Dalpha"][mask], linewidth=1.0)
    axes[3].set_ylabel(r"D$_\alpha$")

    axes[4].plot(tt, sig["P_LH"][mask], linewidth=1.0, label="LH")
    axes[4].plot(tt, sig["P_ECRH"][mask], linewidth=1.0, label="ECRH")
    axes[4].plot(tt, sig["P_NBI"][mask], linewidth=1.0, label="NBI")
    axes[4].set_ylabel("Power")
    axes[4].legend(loc="upper right", frameon=False)

    axes[5].plot(tt, sig["WMHD"][mask], linewidth=1.0)
    axes[5].set_ylabel("WMHD")
    axes[5].set_xlabel("Time (s)")

    if feat is not None:
        t1 = feat["hmode_t1"]
        t2 = feat["hmode_t2"]
        for ax in axes:
            ax.axvspan(t1, t2, alpha=0.15)

    title = f"Shot {shot} | score={score} | label={label}"
    if feat is not None:
        title += (
            f" | q95={feat['q95_mean']:.2f}"
            f" | RF ratio={feat['rf_ratio']:.2f}"
            f" | quiet={feat['quiet_index']:.2f}"
        )
    fig.suptitle(title, y=1.02)

    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def process_one_shot(
    shot: int,
    reader: EASTMDSReader,
    cfg: ScreenConfig,
    signal_map: Dict[str, SignalSpec],
    plot_dir: Optional[Path] = None
) -> Dict[str, object]:
    """
    处理单炮：
    1. 读取基础信号
    2. 重采样到统一时间轴
    3. 构造加热功率
    4. 寻找 H-mode 候选时间窗
    5. 提取特征并评分
    6. 视需要保存 overview 图
    """
    raw = reader.read_all_signals(shot, signal_map)
    t_grid = np.arange(cfg.t_min, cfg.t_max, cfg.dt)

    sig: Dict[str, np.ndarray] = {}
    for name, (t, y) in raw.items():
        sig[name] = resample_to_grid(t, y, t_grid)

    build_heating_signals(sig)

    analysis_mask, analysis_t1, analysis_t2, analysis_mode_used = build_analysis_mask(
        t=t_grid,
        ip=sig["Ip"],
        cfg=cfg
    )

    wins = detect_hmode_windows(
        t=t_grid,
        dalpha=sig["Dalpha"],
        wmhd=sig["WMHD"],
        ne=sig["ne"],
        cfg=cfg,
        analysis_mask=analysis_mask
    )

    if len(wins) == 0:
        row = {
            "shot": shot,
            "has_hmode_candidate": False,
            "analysis_mode": analysis_mode_used,
            "analysis_t1": analysis_t1,
            "analysis_t2": analysis_t2,
            "hmode_t1": np.nan,
            "hmode_t2": np.nan,
            "hmode_duration": np.nan,
            "q95_mean": np.nan,
            "ne_mean": np.nan,
            "rf_ratio": np.nan,
            "quiet_index": np.nan,
            "score": 0,
            "label": "C",
            "error": "",
        }

        if plot_dir is not None and not cfg.plot_only_candidates:
            outpng = plot_dir / f"{shot}_overview.png"
            save_overview_plot(
                shot=shot,
                t=t_grid,
                sig=sig,
                feat=None,
                score=0,
                label="C",
                outpath=outpng,
                cfg=cfg
            )
        return row

    best_win, best_feat, best_score, best_label = choose_best_window(
        t=t_grid, sig=sig, wins=wins, cfg=cfg, analysis_mask=analysis_mask
    )

    row = {
        "shot": shot,
        "has_hmode_candidate": True,
        "analysis_mode": analysis_mode_used,
        "analysis_t1": analysis_t1,
        "analysis_t2": analysis_t2,
        "hmode_t1": best_feat["hmode_t1"],
        "hmode_t2": best_feat["hmode_t2"],
        "hmode_duration": best_feat["hmode_duration"],
        "q95_mean": best_feat["q95_mean"],
        "ne_mean": best_feat["ne_mean"],
        "rf_ratio": best_feat["rf_ratio"],
        "quiet_index": best_feat["quiet_index"],
        "score": best_score,
        "label": best_label,
        "error": "",
    }

    should_plot = False
    if plot_dir is not None:
        if cfg.plot_only_candidates:
            should_plot = best_label in {"A", "B"}
        else:
            should_plot = True

    if should_plot:
        outpng = plot_dir / f"{shot}_overview.png"
        save_overview_plot(
            shot=shot,
            t=t_grid,
            sig=sig,
            feat=best_feat,
            score=best_score,
            label=best_label,
            outpath=outpng,
            cfg=cfg
        )

    return row


def parse_shot_list_from_file(filepath: Path) -> List[int]:
    shots: List[int] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            shots.append(int(s))
    return shots


def batch_screen(
    shots: List[int],
    cfg: ScreenConfig,
    outdir: Path,
    signal_map: Dict[str, SignalSpec],
    host: str
) -> pd.DataFrame:
    """
    批量筛选。
    """
    ensure_dir(outdir)
    plot_dir = outdir / "overview_plots"
    ensure_dir(plot_dir)

    reader = EASTMDSReader(host=host)
    rows: List[Dict[str, object]] = []

    for i, shot in enumerate(shots, start=1):
        logging.info("Processing %d/%d | shot=%d", i, len(shots), shot)
        try:
            row = process_one_shot(
                shot=shot,
                reader=reader,
                cfg=cfg,
                signal_map=signal_map,
                plot_dir=plot_dir
            )
        except Exception as e:
            logging.exception("Shot %d failed", shot)
            row = {
                "shot": shot,
                "has_hmode_candidate": False,
                "hmode_t1": np.nan,
                "hmode_t2": np.nan,
                "hmode_duration": np.nan,
                "q95_mean": np.nan,
                "ne_mean": np.nan,
                "rf_ratio": np.nan,
                "quiet_index": np.nan,
                "score": -1,
                "label": "ERR",
                "error": str(e),
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["score", "shot"], ascending=[False, True], kind="stable")
    out_csv = outdir / "candidate_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logging.info("Saved summary to %s", out_csv)
    return df


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EAST EHO candidate coarse screener")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--shot-file", type=str, help="包含炮号的一列文本文件")
    group.add_argument("--start", type=int, help="起始炮号")

    p.add_argument("--end", type=int, help="结束炮号（含）")
    p.add_argument("--outdir", type=str, required=True, help="输出目录")
    p.add_argument("--host", type=str, default=MDS_HOST, help="MDSplus 服务器地址")

    # 时间范围
    p.add_argument("--tmin", type=float, default=0.0, help="统一时间轴起点")
    p.add_argument("--tmax", type=float, default=10.0, help="统一时间轴终点")
    p.add_argument("--dt", type=float, default=1e-3, help="统一时间步长，默认 1 ms")

    # 作图范围
    p.add_argument("--plot-tmin", type=float, default=None, help="overview 图最小时间")
    p.add_argument("--plot-tmax", type=float, default=None, help="overview 图最大时间")
    p.add_argument(
        "--plot-only-candidates",
        action="store_true",
        help="只绘制 A/B 类候选炮 overview 图"
    )

    # 阈值
    p.add_argument("--q95-low", type=float, default=3.5)
    p.add_argument("--q95-high", type=float, default=5.5)
    p.add_argument("--rf-thr", type=float, default=0.70, help="RF 主导阈值")
    p.add_argument("--quiet-thr", type=float, default=0.70, help="Dalpha quiet index 阈值")
    p.add_argument("--ne-min", type=float, default=2.0e19, help="密度下限")
    p.add_argument("--hmode-min", type=float, default=0.05, help="H-mode 候选最短时长 (s)")
    p.add_argument("--baseline-window", type=float, default=0.10, help="基线滚动窗口 (s)")
    p.add_argument("--dalpha-drop", type=float, default=0.80, help="Dalpha 下降阈值")
    p.add_argument("--wmhd-rise", type=float, default=1.10, help="WMHD 上升阈值")

    p.add_argument("--loglevel", type=str, default="INFO", help="日志等级")

    # 分析窗口策略
    p.add_argument(
        "--analysis-mode",
        type=str,
        default="full",
        choices=["full", "fixed", "ip_stable"],
        help="研究时间窗模式: full/fixed/ip_stable"
    )
    p.add_argument("--analysis-t1", type=float, default=None, help="fixed 模式起点")
    p.add_argument("--analysis-t2", type=float, default=None, help="fixed 模式终点")
    p.add_argument("--ip-stable-frac", type=float, default=0.90, help="ip_stable 阈值比例")
    p.add_argument("--ip-smooth-window", type=float, default=0.05, help="Ip 平滑窗口(s)")

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if args.shot_file is not None:
        shots = parse_shot_list_from_file(Path(args.shot_file))
    else:
        if args.start is None or args.end is None:
            parser.error("使用 --start 时必须同时给出 --end")
        shots = list(range(args.start, args.end + 1))

    if len(shots) == 0:
        raise ValueError("炮号列表为空")

    if args.analysis_mode == "fixed" and (args.analysis_t1 is None or args.analysis_t2 is None):
        parser.error("analysis-mode=fixed 时，必须给出 --analysis-t1 和 --analysis-t2")

    cfg = ScreenConfig(
        t_min=args.tmin,
        t_max=args.tmax,
        dt=args.dt,
        analysis_mode=args.analysis_mode,
        analysis_t1=args.analysis_t1,
        analysis_t2=args.analysis_t2,
        ip_stable_frac=args.ip_stable_frac,
        ip_smooth_window=args.ip_smooth_window,
        baseline_window=args.baseline_window,
        dalpha_drop_ratio=args.dalpha_drop,
        wmhd_rise_ratio=args.wmhd_rise,
        min_hmode_duration=args.hmode_min,
        q95_low=args.q95_low,
        q95_high=args.q95_high,
        rf_ratio_threshold=args.rf_thr,
        quiet_index_threshold=args.quiet_thr,
        ne_min=args.ne_min,
        plot_t_min=args.plot_tmin,
        plot_t_max=args.plot_tmax,
        plot_only_candidates=args.plot_only_candidates,
    )

    outdir = Path(args.outdir)
    df = batch_screen(
        shots=shots,
        cfg=cfg,
        outdir=outdir,
        signal_map=SIGNAL_MAP,
        host=args.host
    )

    print("\n========== Screening finished ==========")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
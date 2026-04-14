#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
密度剖面参数时间演化分析脚本（带RMSE筛选和时间段区分）
Python版本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path
import warnings

from NeFluc import ReflAnalyzer

# ==================== 参数设置 ====================
results_folder = 'D:\MyPythonCodes\83961'
table_path = Path(results_folder) / 'fitting_results.txt'

# 设置RMSE阈值
RMSE_THRESHOLD = 0.3  # RMSE小于等于此值的数据将被保留

# 设置时间段分界点（ECM和EHO的分界时间）
TIME_THRESHOLD = 3.0  # 3.0s之前是ECM，3.0s之后是EHO

# 设置颜色
ECM_COLOR = '#1f77b4'  # 蓝色 - ECM阶段
EHO_COLOR = '#ff7f0e'  # 橙色 - EHO阶段

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出目录
output_dir = Path(results_folder) / 'python_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

# ==================== 辅助函数 ====================
def extract_time_from_filename(filenames):
    """
    从 ne@2910.fig 这类文件名中提取时间
    假设格式: ne@XXXX.fig，其中 XXXX/1000 = Time(秒)
    """
    times = []
    
    for fname in filenames:
        # 提取数字部分
        match = re.search(r'@(\d+)', fname)
        
        if match:
            # 转换为数值并除以1000得到秒
            time_val = int(match.group(1)) / 1000
            times.append(time_val)
        else:
            # 备用方案：尝试从整个文件名提取数字
            num_matches = re.findall(r'\d+', fname)
            if num_matches:
                time_val = int(num_matches[0]) / 1000
                times.append(time_val)
            else:
                times.append(np.nan)  # 提取失败
                warnings.warn(f'无法从文件名 {fname} 提取时间')
    
    # 检查时间是否单调递增
    times_array = np.array(times)
    if not np.all(np.diff(times_array[~np.isnan(times_array)]) > 0):
        warnings.warn('提取的时间不是单调递增的，请检查文件名格式')
    
    return times_array

# ==================== 读取数据 ====================
print('正在读取拟合结果表格...')

# 检查文件是否存在
if not table_path.exists():
    raise FileNotFoundError(f'结果表格不存在: {table_path}')

# 读取数据
results_df = pd.read_csv(table_path, sep='\t')

# 检查是否有成功数据
success_mask = results_df['status'] == 'success'
if not success_mask.any():
    raise ValueError('没有成功的拟合数据')

# 提取成功数据
success_df = results_df[success_mask].copy()

# 从文件名提取时间
print('正在从文件名提取时间信息...')
success_df['time'] = extract_time_from_filename(success_df['filename'].values)

# ==================== 根据RMSE筛选数据 ====================
print(f'应用RMSE筛选（阈值: {RMSE_THRESHOLD:.2f}）...')
valid_mask = success_df['rmse'] <= RMSE_THRESHOLD

if not valid_mask.any():
    warnings.warn(f'没有满足RMSE<={RMSE_THRESHOLD:.2f}的数据，将显示所有成功数据')
    valid_mask = pd.Series(True, index=success_df.index)  # 显示所有数据
    rmse_filter_applied = False
else:
    print(f'找到 {valid_mask.sum()} 个满足RMSE<={RMSE_THRESHOLD:.2f}的数据点（总共 {success_mask.sum()} 个成功数据点）')
    rmse_filter_applied = True

# 筛选数据
filtered_df = success_df[valid_mask].copy()

# 按时间排序
filtered_df = filtered_df.sort_values('time')

# ==================== 根据时间分界点分离数据 ====================
# ECM阶段：t < 3.0s
ecm_mask = filtered_df['time'] < TIME_THRESHOLD
# EHO阶段：t >= 3.0s
eho_mask = filtered_df['time'] >= TIME_THRESHOLD

# ECM数据
ecm_df = filtered_df[ecm_mask]
# EHO数据
eho_df = filtered_df[eho_mask]

# 统计信息
print('\n========== 时间段分布统计 ==========')
print(f'ECM阶段 (t < {TIME_THRESHOLD:.3f}s): {len(ecm_df)} 个数据点')
print(f'EHO阶段 (t >= {TIME_THRESHOLD:.3f}s): {len(eho_df)} 个数据点')
if not ecm_df.empty:
    print(f'ECM时间范围: {ecm_df["time"].min():.3f} - {ecm_df["time"].max():.3f} s')
if not eho_df.empty:
    print(f'EHO时间范围: {eho_df["time"].min():.3f} - {eho_df["time"].max():.3f} s')
print('===================================')

# ==================== 绘制台基密度时间演化图 ====================
fig1, ax1 = plt.subplots(figsize=(12, 8))
fig1.suptitle(f'台基密度演化 - 炮号 {filtered_df["shot"].iloc[0]}', fontsize=16, fontweight='bold')

# 绘制ECM阶段数据
if not ecm_df.empty:
    ax1.scatter(ecm_df['time'], ecm_df['nped'], 
                s=100, color=ECM_COLOR, marker='o', edgecolor='black', linewidth=1.5,
                label=f'ECM (t < {TIME_THRESHOLD:.3f}s)', zorder=3)
    
    # # 添加ECM数据点标签（RMSE值）
    # for idx, row in ecm_df.iterrows():
    #     ax1.text(row['time'], row['nped'], 
    #              f"{row['rmse']:.3f}",
    #              ha='center', va='bottom',
    #              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
    #                                    facecolor='white', 
    #                                    edgecolor=ECM_COLOR,
    #                                    alpha=0.8),
    #              zorder=4)

# 绘制EHO阶段数据
if not eho_df.empty:
    ax1.scatter(eho_df['time'], eho_df['nped'], 
                s=100, color=EHO_COLOR, marker='s', edgecolor='black', linewidth=1.5,
                label=f'EHO (t >= {TIME_THRESHOLD:.3f}s)', zorder=3)
    
    # # 添加EHO数据点标签（RMSE值）
    # for idx, row in eho_df.iterrows():
    #     ax1.text(row['time'], row['nped'], 
    #              f"{row['rmse']:.3f}",
    #              ha='center', va='top',
    #              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
    #                                    facecolor='white', 
    #                                    edgecolor=EHO_COLOR,
    #                                    alpha=0.8),
    #              zorder=4)

# 添加时间分界线
if not filtered_df.empty:
    ax1.axvline(x=TIME_THRESHOLD, color='gray', linestyle='--', 
                linewidth=2, alpha=0.7,
                label=f'分界线 (t={TIME_THRESHOLD:.3f}s)', zorder=2)

# 设置图形属性
ax1.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
ax1.set_ylabel('台基密度 n_ped', fontsize=14, fontweight='bold')

# 设置坐标轴范围
if not filtered_df.empty:
    x_min = filtered_df['time'].min() - 0.01
    x_max = filtered_df['time'].max() + 0.01
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0, filtered_df['nped'].max() * 1.2)

# 添加网格和美化
ax1.grid(True, alpha=0.3)
ax1.set_axisbelow(True)

# 添加图例
ax1.legend(loc='best', fontsize=11)

# 添加阶段标注
if not filtered_df.empty:
    ax1.text((x_min + TIME_THRESHOLD) / 2, filtered_df['nped'].max() * 1.15,
             'ECM阶段', ha='center', fontsize=12, fontweight='bold', color=ECM_COLOR)
    ax1.text((TIME_THRESHOLD + x_max) / 2, filtered_df['nped'].max() * 1.15,
             'EHO阶段', ha='center', fontsize=12, fontweight='bold', color=EHO_COLOR)

# 设置标题（子标题）
if rmse_filter_applied:
    title_str = f'台基密度演化 (RMSE <= {RMSE_THRESHOLD:.2f})'
else:
    title_str = '台基密度演化 (所有成功数据)'
ax1.set_title(title_str, fontsize=14, fontweight='bold', pad=15)

# 保存台基密度图
fig1_path = output_dir / 'nped_evolution_ecm_eho.png'
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f'台基密度演化图已保存: {fig1_path}')

# ==================== 绘制台基宽度时间演化图 ====================
fig2, ax2 = plt.subplots(figsize=(12, 8))
fig2.suptitle(f'台基宽度演化 - 炮号 {filtered_df["shot"].iloc[0]}', fontsize=16, fontweight='bold')

# 绘制ECM阶段数据
if not ecm_df.empty:
    ax2.scatter(ecm_df['time'], ecm_df['nwidth'], 
                s=100, color=ECM_COLOR, marker='o', edgecolor='black', linewidth=1.5,
                label=f'ECM (t < {TIME_THRESHOLD:.3f}s)', zorder=3)
    
    # # 添加ECM数据点标签（RMSE值）
    # for idx, row in ecm_df.iterrows():
    #     ax2.text(row['time'], row['nwidth'], 
    #              f"{row['rmse']:.3f}",
    #              ha='center', va='bottom',
    #              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
    #                                    facecolor='white', 
    #                                    edgecolor=ECM_COLOR,
    #                                    alpha=0.8),
    #              zorder=4)

# 绘制EHO阶段数据
if not eho_df.empty:
    ax2.scatter(eho_df['time'], eho_df['nwidth'], 
                s=100, color=EHO_COLOR, marker='s', edgecolor='black', linewidth=1.5,
                label=f'EHO (t >= {TIME_THRESHOLD:.3f}s)', zorder=3)
    
    # # 添加EHO数据点标签（RMSE值）
    # for idx, row in eho_df.iterrows():
    #     ax2.text(row['time'], row['nwidth'], 
    #              f"{row['rmse']:.3f}",
    #              ha='center', va='top',
    #              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
    #                                    facecolor='white', 
    #                                    edgecolor=EHO_COLOR,
    #                                    alpha=0.8),
    #              zorder=4)

# 添加时间分界线
if not filtered_df.empty:
    ax2.axvline(x=TIME_THRESHOLD, color='gray', linestyle='--', 
                linewidth=2, alpha=0.7,
                label=f'分界线 (t={TIME_THRESHOLD:.3f}s)', zorder=2)

# 设置图形属性
ax2.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
ax2.set_ylabel('ne width  ', fontsize=14, fontweight='bold')

# 设置坐标轴范围
if not filtered_df.empty:
    x_min = filtered_df['time'].min() - 0.01
    x_max = filtered_df['time'].max() + 0.01
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0, filtered_df['nwidth'].max() * 1.2)

# 添加网格和美化
ax2.grid(True, alpha=0.3)
ax2.set_axisbelow(True)

# 添加图例
ax2.legend(loc='best', fontsize=11)

# 添加阶段标注
if not filtered_df.empty:
    ax2.text((x_min + TIME_THRESHOLD) / 2, filtered_df['nwidth'].max() * 1.15,
             'ECM阶段', ha='center', fontsize=12, fontweight='bold', color=ECM_COLOR)
    ax2.text((TIME_THRESHOLD + x_max) / 2, filtered_df['nwidth'].max() * 1.15,
             'EHO阶段', ha='center', fontsize=12, fontweight='bold', color=EHO_COLOR)

# 设置标题（子标题）
if rmse_filter_applied:
    title_str = f'台基宽度演化 (RMSE <= {RMSE_THRESHOLD:.2f})'
else:
    title_str = '台基宽度演化 (所有成功数据)'
ax2.set_title(title_str, fontsize=14, fontweight='bold', pad=15)

# 保存台基宽度图
fig2_path = output_dir / 'nwidth_evolution_ecm_eho.png'
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f'台基宽度演化图已保存: {fig2_path}')

# ==================== 绘制时间段分析统计图 ====================
fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle(f'时间段分析统计 - 炮号 {filtered_df["shot"].iloc[0]}', 
              fontsize=16, fontweight='bold', y=1.02)

# 子图1：ECM和EHO的RMSE分布对比
ax3_1 = axes[0]

if not ecm_df.empty and not eho_df.empty:
    # 绘制ECM和EHO的RMSE分布
    bins = np.linspace(min(filtered_df['rmse'].min(), RMSE_THRESHOLD), 
                       filtered_df['rmse'].max(), 20)
    ax3_1.hist(ecm_df['rmse'], bins=bins, alpha=0.7, color=ECM_COLOR, 
               edgecolor='black', label=f'ECM (n={len(ecm_df)})')
    ax3_1.hist(eho_df['rmse'], bins=bins, alpha=0.7, color=EHO_COLOR, 
               edgecolor='black', label=f'EHO (n={len(eho_df)})')
    
    # 添加阈值线
    ax3_1.axvline(x=RMSE_THRESHOLD, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'阈值={RMSE_THRESHOLD:.2f}')
    
    ax3_1.legend(loc='best')
elif not ecm_df.empty:
    ax3_1.hist(ecm_df['rmse'], bins=20, alpha=0.7, color=ECM_COLOR, 
               edgecolor='black', label=f'ECM (n={len(ecm_df)})')
    ax3_1.legend(loc='best')
elif not eho_df.empty:
    ax3_1.hist(eho_df['rmse'], bins=20, alpha=0.7, color=EHO_COLOR, 
               edgecolor='black', label=f'EHO (n={len(eho_df)})')
    ax3_1.legend(loc='best')
else:
    ax3_1.text(0.5, 0.5, '无数据', ha='center', va='center', 
               fontsize=12, fontweight='bold', transform=ax3_1.transAxes)

ax3_1.set_xlabel('RMSE值', fontsize=12, fontweight='bold')
ax3_1.set_ylabel('数据点数', fontsize=12, fontweight='bold')
ax3_1.set_title('ECM和EHO的RMSE分布', fontsize=14, fontweight='bold')
ax3_1.grid(True, alpha=0.3)

# 子图2：ECM阶段统计
ax3_2 = axes[1]

if not ecm_df.empty:
    # 创建条形图显示ECM统计信息
    bar_labels = ['n_ped', 'n_width', 'RMSE']
    bar_values = [
        ecm_df['nped'].mean(),
        ecm_df['nwidth'].mean(),
        ecm_df['rmse'].mean()
    ]
    
    bars = ax3_2.bar(range(len(bar_labels)), bar_values, color=ECM_COLOR, alpha=0.8)
    ax3_2.set_xticks(range(len(bar_labels)))
    ax3_2.set_xticklabels(bar_labels, fontsize=11)
    ax3_2.set_ylabel('平均值', fontsize=12, fontweight='bold')
    ax3_2.set_title(f'ECM阶段统计 (n={len(ecm_df)})', fontsize=14, fontweight='bold')
    ax3_2.grid(True, alpha=0.3, axis='y')
    ax3_2.set_ylim(0, 4.5)
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, bar_values)):
        height = bar.get_height()
        ax3_2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
else:
    ax3_2.text(0.5, 0.5, '无ECM阶段数据', ha='center', va='center', 
               fontsize=12, fontweight='bold', transform=ax3_2.transAxes)

# 子图3：EHO阶段统计
ax3_3 = axes[2]

if not eho_df.empty:
    # 创建条形图显示EHO统计信息
    bar_labels = ['n_ped', 'n_width', 'RMSE']
    bar_values = [
        eho_df['nped'].mean(),
        eho_df['nwidth'].mean(),
        eho_df['rmse'].mean()
    ]
    
    bars = ax3_3.bar(range(len(bar_labels)), bar_values, color=EHO_COLOR, alpha=0.8)
    ax3_3.set_xticks(range(len(bar_labels)))
    ax3_3.set_xticklabels(bar_labels, fontsize=11)
    ax3_3.set_ylabel('平均值', fontsize=12, fontweight='bold')
    ax3_3.set_title(f'EHO阶段统计 (n={len(eho_df)})', fontsize=14, fontweight='bold')
    ax3_3.grid(True, alpha=0.3, axis='y')
    ax3_3.set_ylim(0, 4.5)
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, bar_values)):
        height = bar.get_height()
        ax3_3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
else:
    ax3_3.text(0.5, 0.5, '无EHO阶段数据', ha='center', va='center', 
               fontsize=12, fontweight='bold', transform=ax3_3.transAxes)

# 调整布局
plt.tight_layout()

# 保存时间段分析统计图
fig3_path = output_dir / 'time_period_analysis_statistics.png'
fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f'时间段分析统计图已保存: {fig3_path}')

# ==================== 保存详细数据到CSV ====================
# 保存筛选后的数据
filtered_data_path = output_dir / 'filtered_data.csv'
filtered_df.to_csv(filtered_data_path, index=False, encoding='utf-8-sig')
print(f'筛选后的数据已保存: {filtered_data_path}')

# 保存ECM和EHO分开的数据
if not ecm_df.empty:
    ecm_data_path = output_dir / 'ecm_data.csv'
    ecm_df.to_csv(ecm_data_path, index=False, encoding='utf-8-sig')
    print(f'ECM阶段数据已保存: {ecm_data_path}')

if not eho_df.empty:
    eho_data_path = output_dir / 'eho_data.csv'
    eho_df.to_csv(eho_data_path, index=False, encoding='utf-8-sig')
    print(f'EHO阶段数据已保存: {eho_data_path}')

# ==================== 输出统计信息 ====================
print('\n' + '=' * 40 + ' 详细统计信息 ' + '=' * 40)
print(f'炮号: {filtered_df["shot"].iloc[0]}')
print(f'总成功数据点: {len(success_df)}')
print(f'应用RMSE筛选 (<= {RMSE_THRESHOLD:.2f}): {"是" if rmse_filter_applied else "否"}')
print(f'筛选后数据点: {len(filtered_df)}')

if not filtered_df.empty:
    print(f'时间范围: {filtered_df["time"].min():.3f} - {filtered_df["time"].max():.3f} s')
    
    print(f'\n--- ECM阶段 (t < {TIME_THRESHOLD:.3f}s) ---')
    if not ecm_df.empty:
        print(f'数据点数: {len(ecm_df)}')
        print(f'时间范围: {ecm_df["time"].min():.3f} - {ecm_df["time"].max():.3f} s')
        print(f'台基密度范围: {ecm_df["nped"].min():.3f} - {ecm_df["nped"].max():.3f}')
        print(f'台基密度平均值: {ecm_df["nped"].mean():.3f} ± {ecm_df["nped"].std():.3f}')
        print(f'台基宽度范围: {ecm_df["nwidth"].min():.3f} - {ecm_df["nwidth"].max():.3f}')
        print(f'台基宽度平均值: {ecm_df["nwidth"].mean():.3f} ± {ecm_df["nwidth"].std():.3f}')
        print(f'RMSE范围: {ecm_df["rmse"].min():.3f} - {ecm_df["rmse"].max():.3f}')
        print(f'RMSE平均值: {ecm_df["rmse"].mean():.3f} ± {ecm_df["rmse"].std():.3f}')
    else:
        print('无数据')
    
    print(f'\n--- EHO阶段 (t >= {TIME_THRESHOLD:.3f}s) ---')
    if not eho_df.empty:
        print(f'数据点数: {len(eho_df)}')
        print(f'时间范围: {eho_df["time"].min():.3f} - {eho_df["time"].max():.3f} s')
        print(f'台基密度范围: {eho_df["nped"].min():.3f} - {eho_df["nped"].max():.3f}')
        print(f'台基密度平均值: {eho_df["nped"].mean():.3f} ± {eho_df["nped"].std():.3f}')
        print(f'台基宽度范围: {eho_df["nwidth"].min():.3f} - {eho_df["nwidth"].max():.3f}')
        print(f'台基宽度平均值: {eho_df["nwidth"].mean():.3f} ± {eho_df["nwidth"].std():.3f}')
        print(f'RMSE范围: {eho_df["rmse"].min():.3f} - {eho_df["rmse"].max():.3f}')
        print(f'RMSE平均值: {eho_df["rmse"].mean():.3f} ± {eho_df["rmse"].std():.3f}')
    else:
        print('无数据')
else:
    print('筛选后无有效数据')

print('=' * 95)

# ================================结合频频谱分析结果输出============================
# --- 配置 ---
shot_num = filtered_df["shot"].iloc[0]   # 修改为真实存在的 Shot
t_start = x_min = filtered_df['time'].min() - 0.01
t_end = x_max = filtered_df['time'].max() + 0.01

# 请确保此路径下有数据 (Z盘需要挂载)
# 如果在本地测试，请修改 _setup_config 中的路径

print("=== 开始运行对比示例 ===")

# 1. 获取 Judge = 1 (High Res / Original) 的数据
analyzer1 = ReflAnalyzer(shot_num, t_start, t_end, judge=1)
t1, f1, psds1, freqs1 = analyzer1.run()

# 2. 获取 Judge = 2 (Amplitude / Filtered) 的数据
analyzer2 = ReflAnalyzer(shot_num, t_start, t_end, judge=2)
t2, f2, psds2, freqs2 = analyzer2.run()

fig_4, axe_4 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
# 密度涨落原始数据
axe_4[0].set_title('密度涨落原始数据', fontsize=12, fontweight='bold')
freq_val = freqs1[3]  # 选择一个频率索引进行展示
title_l = f"#{shot_num} {freq_val}GHz (Original)"
ReflAnalyzer.plot_on_axis(axe_4[0], t1, f1, psds1[3], title_l)

# 密度涨落幅度信号
axe_4[1].set_title('密度涨落幅度信号', fontsize=12, fontweight='bold')
title_l = f"#{shot_num} {freq_val}GHz (Filtered)"
ReflAnalyzer.plot_on_axis(axe_4[1], t2, f2, psds2[3], title_l)
# 台基密度时间演化
if not ecm_df.empty:
    axe_4[2].scatter(ecm_df['time'], ecm_df['nped'], 
                s=100, color=ECM_COLOR, marker='o', edgecolor='black', linewidth=1.5,
                label=f'ECM (t < {TIME_THRESHOLD:.3f}s)', zorder=3)
    
    # # 添加ECM数据点标签（RMSE值）
    # for idx, row in ecm_df.iterrows():
    #     ax1.text(row['time'], row['nped'], 
    #              f"{row['rmse']:.3f}",
    #              ha='center', va='bottom',
    #              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
    #                                    facecolor='white', 
    #                                    edgecolor=ECM_COLOR,
    #                                    alpha=0.8),
    #              zorder=4)

# 绘制EHO阶段数据
if not eho_df.empty:
    axe_4[2].scatter(eho_df['time'], eho_df['nped'], 
                s=100, color=EHO_COLOR, marker='s', edgecolor='black', linewidth=1.5,
                label=f'EHO (t >= {TIME_THRESHOLD:.3f}s)', zorder=3)
    
    # # 添加EHO数据点标签（RMSE值）
    # for idx, row in eho_df.iterrows():
    #     ax1.text(row['time'], row['nped'], 
    #              f"{row['rmse']:.3f}",
    #              ha='center', va='top',
    #              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
    #                                    facecolor='white', 
    #                                    edgecolor=EHO_COLOR,
    #                                    alpha=0.8),
    #              zorder=4)

# 添加时间分界线
if not filtered_df.empty:
    axe_4[2].axvline(x=TIME_THRESHOLD, color='gray', linestyle='--', 
                linewidth=2, alpha=0.7,
                label=f'分界线 (t={TIME_THRESHOLD:.3f}s)', zorder=2)

# 设置图形属性
# axe_4[2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axe_4[2].set_ylabel('台基密度 n_ped', fontsize=12, fontweight='bold')

# 设置坐标轴范围
if not filtered_df.empty:
    x_min = filtered_df['time'].min() - 0.01
    x_max = filtered_df['time'].max() + 0.01
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0, filtered_df['nped'].max() * 1.2)

# 添加网格和美化
axe_4[2].grid(True, alpha=0.3)
axe_4[2].set_axisbelow(True)

# 添加图例
axe_4[2].legend(loc='best', fontsize=11)
# 添加阶段标注
if not filtered_df.empty:
    axe_4[2].text((x_min + TIME_THRESHOLD) / 2, filtered_df['nped'].max() * 1.15,
             'ECM阶段', ha='center', fontsize=12, fontweight='bold', color=ECM_COLOR)
    axe_4[2].text((TIME_THRESHOLD + x_max) / 2, filtered_df['nped'].max() * 1.15,
             'EHO阶段', ha='center', fontsize=12, fontweight='bold', color=EHO_COLOR)

# 设置标题（子标题）
if rmse_filter_applied:
    title_str = f'台基密度演化 (RMSE <= {RMSE_THRESHOLD:.2f})'
else:
    title_str = '台基密度演化 (所有成功数据)'
axe_4[2].set_title(title_str, fontsize=12, fontweight='bold')
# ================台基宽度时间演化==================
if not ecm_df.empty:
    axe_4[3].scatter(ecm_df['time'], ecm_df['nwidth'], 
                s=100, color=ECM_COLOR, marker='o', edgecolor='black', linewidth=1.5,
                label=f'ECM (t < {TIME_THRESHOLD:.3f}s)', zorder=3)
    
    # # 添加ECM数据点标签（RMSE值）
    # for idx, row in ecm_df.iterrows():
    #     ax2.text(row['time'], row['nwidth'], 
    #              f"{row['rmse']:.3f}",
    #              ha='center', va='bottom',
    #              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
    #                                    facecolor='white', 
    #                                    edgecolor=ECM_COLOR,
    #                                    alpha=0.8),
    #              zorder=4)
# 绘制EHO阶段数据
if not eho_df.empty:
    axe_4[3].scatter(eho_df['time'], eho_df['nwidth'], 
                s=100, color=EHO_COLOR, marker='s', edgecolor='black', linewidth=1.5,
                label=f'EHO (t >= {TIME_THRESHOLD:.3f}s)', zorder=3)
    
    # # 添加EHO数据点标签（RMSE值）
    # for idx, row in eho_df.iterrows():
    #     ax2.text(row['time'], row['nwidth'], 
    #              f"{row['rmse']:.3f}",
    #              ha='center', va='top',
    #              fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
    #                                    facecolor='white', 
    #                                    edgecolor=EHO_COLOR,
    #                                    alpha=0.8),
    #              zorder=4)
# 添加时间分界线
if not filtered_df.empty:   
    axe_4[3].axvline(x=TIME_THRESHOLD, color='gray', linestyle='--', 
                linewidth=2, alpha=0.7,
                label=f'分界线 (t={TIME_THRESHOLD:.3f}s)', zorder=2)
# 设置图形属性
# axe_4[3].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axe_4[3].set_ylabel('ne width  ', fontsize=12, fontweight='bold')
# 设置坐标轴范围
if not filtered_df.empty:
    x_min = filtered_df['time'].min() - 0.01
    x_max = filtered_df['time'].max() + 0.01
    axe_4[3].set_xlim(x_min, x_max)
    axe_4[3].set_ylim(0, filtered_df['nwidth'].max() * 1.2)
# 添加网格和美化
axe_4[3].grid(True, alpha=0.3)
axe_4[3].set_axisbelow(True)

# 添加图例
axe_4[3].legend(loc='best', fontsize=11)
# 添加阶段标注
if not filtered_df.empty:
    axe_4[3].text((x_min + TIME_THRESHOLD) / 2, filtered_df['nwidth'].max() * 1.15,
             'ECM阶段', ha='center', fontsize=12, fontweight='bold', color=ECM_COLOR)
    axe_4[3].text((TIME_THRESHOLD + x_max) / 2, filtered_df['nwidth'].max() * 1.15,
             'EHO阶段', ha='center', fontsize=12, fontweight='bold', color=EHO_COLOR)
# 设置标题（子标题）
if rmse_filter_applied:
    title_str = f'台基宽度演化 (RMSE <= {RMSE_THRESHOLD:.2f})'
else:
    title_str = '台基宽度演化 (所有成功数据)'
axe_4[3].set_title(title_str, fontsize=12, fontweight='bold')

fig4_path = output_dir / 'Fluc_time_period_analysis_statistics.png'
fig_4.savefig(fig4_path, dpi=300, bbox_inches='tight')
print(f'时间段分析统计图已保存: {fig4_path}')
# 显示所有图形
plt.show()

print(f'\n分析完成！所有结果已保存到: {output_dir}')
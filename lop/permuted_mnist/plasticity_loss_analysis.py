# -*- coding: utf-8 -*-
# 仅保留第一个子图 + 线性直线虚线 + 更换颜色
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== 配置 =====================
CSV_FILE_PATH = "0/0_0.csv"
SAVE_IMAGE_PATH = "./only_training_iters_plot.png"
DPI = 300

# 字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ===================== 读取数据 =====================
df = pd.read_csv(CSV_FILE_PATH)

# ===================== 计算每个任务的总迭代数 / 3000 =====================
task_metrics = []
unique_tasks = sorted(df['task_idx'].unique())

for task in unique_tasks:
    task_data = df[df['task_idx'] == task].sort_values('iter')
    if len(task_data) < 2:
        continue
    
    first = task_data['iter'].iloc[0]
    last = task_data['iter'].iloc[-1]
    total = (last - first) / 3000  # 👈 按你要求除以3000
    
    task_metrics.append({
        'task_idx': task,
        'total_iters': total
    })

metrics_df = pd.DataFrame(task_metrics)
x = metrics_df['task_idx'].values
y = metrics_df['total_iters'].values

# ===================== 绘制图像 =====================
plt.figure(figsize=(10, 6))

# 1. 原始数据实线（蓝色）
plt.plot(x, y, color='#2E86AB', linewidth=2.5, marker='o', markersize=3, alpha=0.8, label='Training Iterations')

# 2. 线性拟合虚线（换色：深红色）
z1 = np.polyfit(x, y, 1)
p1 = np.poly1d(z1)
plt.plot(x, p1(x), color='#C73E1D', linewidth=2.5, linestyle='--', 
         alpha=0.9, label=f'Linear Trend (Slope: {z1[0]:.2f})')

# ===================== 其余不变 =====================
plt.title('Training Iterations per Task (Divided by 3000)', fontsize=16, fontweight='bold')
plt.xlabel('Task Index', fontsize=12)
plt.ylabel('Training Iterations (/3000)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(0, metrics_df['task_idx'].max())

plt.tight_layout()
plt.savefig(SAVE_IMAGE_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 已生成：直线虚线 + 新颜色版本！")
print(f"📊 图片保存路径：{os.path.abspath(SAVE_IMAGE_PATH)}")
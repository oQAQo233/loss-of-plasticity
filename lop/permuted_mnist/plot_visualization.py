import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ===================== 修复字体警告 + 论文级绘图风格 =====================
plt.rcParams["font.family"] = "DejaVu Sans"  # 替换Times New Roman为服务器存在的字体
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

# ===================== 路径配置（适配你的目录结构） =====================
# 获取当前脚本所在目录（确保路径绝对正确）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_DIR = os.path.join(CURRENT_DIR, "preprocessed_data")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "paper_figures")
JOINT_DIR = os.path.join(CURRENT_DIR, "2", "joint_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 论文级配色方案 =====================
COLORS = {
    "continual": "#2E86AB",    # 蓝色（持续学习）
    "independent": "#A23B72",  # 紫红色（独立学习）
    "joint_per_task": "#F18F01",  # 橙色（联合训练 per-task）
    "joint_final": "#C73E1D"       # 深红色（联合训练 final）
}

# ===================== 核心绘图函数 =====================
def plot_continual_acc_heatmap():
    """绘制持续学习准确率热力图（灾难性遗忘核心图）"""
    acc_matrix_path = os.path.join(PREPROCESS_DIR, "continual_acc_matrix.csv")
    if not os.path.exists(acc_matrix_path):
        raise FileNotFoundError(f"准确率矩阵文件不存在：{acc_matrix_path}")
    
    acc_matrix_df = pd.read_csv(acc_matrix_path, index_col=0)
    acc_matrix = acc_matrix_df.values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # 绘制热力图（viridis配色，0-1范围）
    im = ax.imshow(acc_matrix, cmap="viridis", vmin=0, vmax=1)
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=14)
    
    # 设置坐标轴（适配任务数）
    n_tasks = acc_matrix.shape[0]
    ax.set_xticks(np.arange(n_tasks))
    ax.set_yticks(np.arange(n_tasks))
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)], rotation=45)
    ax.set_yticklabels([f"After T{i}" for i in range(n_tasks)])
    
    # 添加标题和标签
    ax.set_title("Catastrophic Forgetting in Continual Learning", fontsize=16, pad=20)
    ax.set_xlabel("Test Task", fontsize=14)
    ax.set_ylabel("After Training Task", fontsize=14)
    
    # 优化布局
    plt.tight_layout()
    # 保存（PDF矢量图+PNG位图）
    plt.savefig(os.path.join(OUTPUT_DIR, "continual_acc_heatmap.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "continual_acc_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("✅ 准确率热力图已保存")

def plot_nc_time_series():
    """绘制NC1~NC4时间序列曲线（修复联合训练文件读取 + 双联合曲线 + 整数横轴）"""
    # 1. 读取持续/独立学习NC数据
    continual_final_path = os.path.join(PREPROCESS_DIR, "continual_final_nc.csv")
    independent_final_path = os.path.join(PREPROCESS_DIR, "independent_final_nc.csv")
    
    if not os.path.exists(continual_final_path):
        raise FileNotFoundError(f"持续学习NC文件不存在：{continual_final_path}")
    if not os.path.exists(independent_final_path):
        raise FileNotFoundError(f"独立学习NC文件不存在：{independent_final_path}")
    
    continual_final = pd.read_csv(continual_final_path)
    independent_final = pd.read_csv(independent_final_path)
    task_idx = continual_final["task_idx"].values
    n_tasks = len(task_idx)
    
    # 2. 自动匹配联合训练文件（不硬编码任务数）
    # 匹配 joint_model_per_task_accuracy_*.csv
    joint_per_task_files = [f for f in os.listdir(JOINT_DIR) if re.match(r"joint_model_per_task_accuracy_\d+_tasks\.csv", f)]
    if not joint_per_task_files:
        raise FileNotFoundError(f"在 {JOINT_DIR} 下未找到 joint_model_per_task_accuracy_*_tasks.csv 文件")
    joint_per_task_file = sorted(joint_per_task_files)[-1]  # 取最新的
    joint_per_task_path = os.path.join(JOINT_DIR, joint_per_task_file)
    joint_per_task = pd.read_csv(joint_per_task_path)
    
    # 匹配 joint_training_results.csv
    joint_final_file = "joint_training_results.csv"
    joint_final_path = os.path.join(JOINT_DIR, joint_final_file)
    if not os.path.exists(joint_final_path):
        raise FileNotFoundError(f"联合训练结果文件不存在：{joint_final_path}")
    joint_final = pd.read_csv(joint_final_path)
    
    # 提取NC指标
    nc_cols = ["nc1", "nc2", "nc3", "nc4"]
    
    # 绘制2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, nc_col in enumerate(nc_cols):
        ax = axes[idx]
        # 绘制持续学习曲线
        ax.plot(task_idx, continual_final[nc_col].values, 
                label="Continual", color=COLORS["continual"], linewidth=2, marker="o", markersize=4)
        # 绘制独立学习曲线
        ax.plot(task_idx, independent_final[nc_col].values, 
                label="Independent", color=COLORS["independent"], linewidth=2, marker="s", markersize=4)
        # 绘制联合训练 per-task 曲线
        joint_per_task_vals = joint_per_task[nc_col].values[:n_tasks]
        if len(joint_per_task_vals) < n_tasks:
            joint_per_task_vals = np.pad(joint_per_task_vals, (0, n_tasks - len(joint_per_task_vals)), mode="edge")
        ax.plot(task_idx, joint_per_task_vals, 
                label="Joint (per-task)", color=COLORS["joint_per_task"], linewidth=2, linestyle="-", marker="^", markersize=4)
        # 绘制联合训练 final 曲线
        joint_final_val = joint_final[nc_col].iloc[0]
        joint_final_vals = np.full(shape=task_idx.shape, fill_value=joint_final_val, dtype=np.float64)
        # 调试打印
        print(f"[绘图调试] nc_col={nc_col}, task_idx={task_idx}, joint_final_vals={joint_final_vals}")
        ax.plot(task_idx, joint_final_vals, 
                label="Joint (final)", color=COLORS["joint_final"], linewidth=2, linestyle="--", marker="D", markersize=4)
        
        # ========== 核心修改：自定义纵轴范围 ==========
        # 收集所有曲线的数值，计算全局最大/最小值
        all_values = np.concatenate([
            continual_final[nc_col].values,
            independent_final[nc_col].values,
            joint_per_task_vals,
            joint_final_vals
        ])
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        
        # 计算上下余量（10%），避免数值贴边
        margin = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.1
        y_min = min_val - margin
        y_max = max_val + margin
        
        # 设置纵轴范围（如果最小值-余量为负，强制从0开始，避免负数）
        ax.set_ylim(bottom=max(y_min, 0), top=y_max)
        # =============================================
        
        # 设置子图样式
        ax.set_title(f"{nc_col.upper()} Evolution", fontsize=14)
        ax.set_xlabel("Task Index", fontsize=12)
        ax.set_ylabel(nc_col.upper(), fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        # 强制整数横轴
        ax.set_xticks(np.arange(0, max(task_idx)+1, 1))
        ax.set_xticklabels([str(int(x)) for x in np.arange(0, max(task_idx)+1, 1)])
        ax.set_xlim(0, max(task_idx))
    
    # 整体标题
    fig.suptitle("Neural Collapse (NC) Metrics Across Tasks", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # 保存
    plt.savefig(os.path.join(OUTPUT_DIR, "nc_time_series.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "nc_time_series.png"), bbox_inches="tight")
    plt.close()
    print("✅ NC指标时间序列图已保存（含2条联合训练曲线 + 整数横轴）")

def plot_dead_neuron_heatmap():
    """绘制死亡神经元比例热力图"""
    continual_dead_path = os.path.join(PREPROCESS_DIR, "continual_dead_ratio.csv")
    if not os.path.exists(continual_dead_path):
        raise FileNotFoundError(f"死亡神经元文件不存在：{continual_dead_path}")
    
    continual_dead = pd.read_csv(continual_dead_path)
    dead_matrix = continual_dead.values
    n_tasks = dead_matrix.shape[0]
    n_layers = dead_matrix.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # 绘制热力图（Reds配色，0-1范围）
    im = ax.imshow(dead_matrix.T, cmap="Reds", vmin=0, vmax=1)
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Dead Neuron Ratio", fontsize=14)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(n_tasks))
    ax.set_yticks(np.arange(n_layers))
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)], rotation=45)
    ax.set_yticklabels([f"Layer {i+1}" for i in range(n_layers)])
    
    # 添加标题和标签
    ax.set_title("Dead Neuron Ratio in Continual Learning", fontsize=16, pad=20)
    ax.set_xlabel("Task Index", fontsize=14)
    ax.set_ylabel("Hidden Layer", fontsize=14)
    
    plt.tight_layout()
    # 保存
    plt.savefig(os.path.join(OUTPUT_DIR, "dead_neuron_heatmap.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "dead_neuron_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("✅ 死亡神经元热力图已保存")

def plot_termination_pie():
    """绘制训练终止条件饼图"""
    termination_path = os.path.join(PREPROCESS_DIR, "termination_stats.csv")
    if not os.path.exists(termination_path):
        raise FileNotFoundError(f"终止条件文件不存在：{termination_path}")
    
    termination_df = pd.read_csv(termination_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    labels = ["Reached Threshold (0.96)", "Epoch Exhausted"]
    colors_pie = ["#60A5FA", "#F87171"]  # 蓝/红配色
    
    for idx, paradigm in enumerate(["Continual", "Independent"]):
        ax = axes[idx]
        data = termination_df[termination_df["Paradigm"] == paradigm]
        sizes = [data["Reached_Threshold"].values[0], data["Epoch_Exhausted"].values[0]]
        # 绘制饼图
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, 
                                          autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{paradigm} Training Termination", fontsize=14)
        # 美化文字（白色填充）
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(10)
    
    plt.tight_layout()
    # 保存
    plt.savefig(os.path.join(OUTPUT_DIR, "termination_pie.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "termination_pie.png"), bbox_inches="tight")
    plt.close()
    print("✅ 终止条件饼图已保存")

def plot_paradigm_comparison():
    """绘制三种训练范式对比面板（柱状图）"""
    compare_path = os.path.join(PREPROCESS_DIR, "paradigm_comparison.csv")
    if not os.path.exists(compare_path):
        raise FileNotFoundError(f"范式对比文件不存在：{compare_path}")
    
    compare_df = pd.read_csv(compare_path)
    paradigms = compare_df["Training_Paradigm"].values
    
    # 选择核心对比指标
    metrics = ["Average_Accuracy", "Average_NC1", "Avg_Dead_Neuron_Ratio"]
    metric_labels = ["Average Accuracy", "Average NC1", "Avg Dead Neuron Ratio"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        # 提取数值
        values = compare_df[metric].values
        # 处理联合训练死亡神经元缺失值
        if metric == "Avg_Dead_Neuron_Ratio":
            values[2] = np.nan  # Joint无数据
        
        # 绘制柱状图
        bars = ax.bar(paradigms, values, 
                      color=[COLORS["continual"], COLORS["independent"], COLORS["joint_per_task"]],
                      width=0.6, edgecolor="black", linewidth=1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f"{height:.3f}", ha="center", va="bottom", fontsize=10)
        
        # 设置子图样式
        ax.set_title(label, fontsize=14)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        # 适配y轴范围
        if metric == "Average_Accuracy":
            ax.set_ylim(0, 1.0)
        elif metric == "Average_NC1":
            ax.set_ylim(0, max(values) * 1.2)
        elif metric == "Avg_Dead_Neuron_Ratio":
            ax.set_ylim(0, max(values[:2]) * 1.2)
    
    # 整体标题
    fig.suptitle("Performance Comparison Across Training Paradigms", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # 保存
    plt.savefig(os.path.join(OUTPUT_DIR, "paradigm_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "paradigm_comparison.png"), bbox_inches="tight")
    plt.close()
    print("✅ 训练范式对比图已保存")

# ===================== 主函数（一键生成所有图表） =====================
def main():
    print("🚀 开始生成论文可视化图表...")
    
    # 检查预处理数据目录是否存在
    if not os.path.exists(PREPROCESS_DIR):
        raise FileNotFoundError(f"预处理数据目录不存在：{PREPROCESS_DIR}\n请先运行：python data_preprocess.py")
    
    # 依次绘制所有图表
    plot_continual_acc_heatmap()
    plot_nc_time_series()
    plot_dead_neuron_heatmap()
    plot_termination_pie()
   # plot_paradigm_comparison()
    
    print("\n🎉 所有可视化图表生成完成！")
    print(f"📁 图表保存路径：{OUTPUT_DIR}/")
    print("📋 生成的图表清单：")
    print("  1. continual_acc_heatmap.pdf/png (灾难性遗忘核心图)")
    print("  2. nc_time_series.pdf/png (NC指标演化，含2条联合训练曲线 + 整数横轴)")
    print("  3. dead_neuron_heatmap.pdf/png (死亡神经元分布)")
    print("  4. termination_pie.pdf/png (训练终止条件)")
    #print("  5. paradigm_comparison.pdf/png (三种范式对比)")

if __name__ == "__main__":
    main()
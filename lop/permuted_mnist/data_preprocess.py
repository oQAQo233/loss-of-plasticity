import os
import re
import pandas as pd
import numpy as np

# 配置主实验目录（和你原有代码一致）
MAIN_DIR = "0/"
JOINT_DIR = os.path.join(MAIN_DIR, "joint_results")
INDEPENDENT_DIR = MAIN_DIR

def load_continual_forward_test():
    """加载持续学习前向测试结果（灾难性遗忘核心数据）"""
    file_path = os.path.join(MAIN_DIR, "continual_forward_test_results.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"前向测试文件不存在：{file_path}")
    
    df = pd.read_csv(file_path)
    # 构建准确率矩阵（行：训练完成的任务，列：测试任务）
    max_train_task = df["current_train_task_idx"].max()
    max_test_task = df["test_task_idx"].max()
    acc_matrix = np.zeros((max_train_task + 1, max_test_task + 1))
    
    for idx, row in df.iterrows():
        i = int(row["current_train_task_idx"])
        j = int(row["test_task_idx"])
        acc_matrix[i, j] = row["accuracy"]
    
    # 保存整理后的矩阵
    matrix_df = pd.DataFrame(acc_matrix)
    matrix_df.columns = [f"Test_Task_{i}" for i in range(acc_matrix.shape[1])]
    matrix_df.index = [f"Train_Task_{i}" for i in range(acc_matrix.shape[0])]
    
    return df, matrix_df

def load_nc_time_series():
    """加载NC指标时间序列（持续/独立/联合训练）"""
    # 1. 持续学习NC数据
    continual_inter = pd.read_csv(os.path.join(MAIN_DIR, "0_0.csv"))
    continual_final = pd.read_csv(os.path.join(MAIN_DIR, "0.csv"))
    
    # 2. 独立学习NC数据
    independent_inter = pd.read_csv(os.path.join(INDEPENDENT_DIR, "independent_intermediate_results.csv"))
    independent_final = pd.read_csv(os.path.join(INDEPENDENT_DIR, "independent_final_results.csv"))
    
    # 3. 联合学习NC数据
    joint_inter = pd.read_csv(os.path.join(JOINT_DIR, "joint_intermediate_results.csv"))
    joint_final = pd.read_csv(os.path.join(JOINT_DIR, "joint_training_results.csv"))
    
    return {
        "continual_inter": continual_inter,
        "continual_final": continual_final,
        "independent_inter": independent_inter,
        "independent_final": independent_final,
        "joint_inter": joint_inter,
        "joint_final": joint_final
    }

def load_dead_neurons():
    """加载死亡神经元数据"""
    # 持续学习死亡神经元
    continual_dead = pd.read_csv(os.path.join(MAIN_DIR, "0.csv"))
    # 独立学习死亡神经元
    independent_dead = pd.read_csv(os.path.join(INDEPENDENT_DIR, "independent_final_results.csv"))
    
    # 提取死亡神经元列（适配不同隐藏层数）
    dead_cols = [col for col in continual_dead.columns if "dead_neurons_" in col]
    continual_dead_matrix = continual_dead[dead_cols].values
    independent_dead_matrix = independent_dead[dead_cols].values
    
    # 计算死亡神经元比例（假设每层神经元数固定为200，可根据你的模型调整）
    neuron_per_layer = 200
    continual_dead_ratio = continual_dead_matrix / neuron_per_layer
    independent_dead_ratio = independent_dead_matrix / neuron_per_layer
    
    return {
        "continual_dead_ratio": pd.DataFrame(continual_dead_ratio, columns=dead_cols),
        "independent_dead_ratio": pd.DataFrame(independent_dead_ratio, columns=dead_cols),
        "continual_task_idx": continual_dead["task_idx"].values
    }

def load_termination_stats():
    """加载训练终止条件统计"""
    # 持续学习终止统计
    continual_terminate = pd.read_csv(os.path.join(MAIN_DIR, "0.csv"))
    # 独立学习终止统计
    independent_terminate = pd.read_csv(os.path.join(INDEPENDENT_DIR, "independent_final_results.csv"))
    
    # 统计阈值达成/轮数耗尽的数量
    continual_stats = {
        "reached_threshold": continual_terminate["reached_threshold"].sum(),
        "epoch_exhausted": len(continual_terminate) - continual_terminate["reached_threshold"].sum(),
        "total_tasks": len(continual_terminate)
    }
    
    independent_stats = {
        "reached_threshold": independent_terminate["reached_threshold"].sum(),
        "epoch_exhausted": len(independent_terminate) - independent_terminate["reached_threshold"].sum(),
        "total_tasks": len(independent_terminate)
    }
    
    return continual_stats, independent_stats

def load_paradigm_comparison():
    """加载三种训练范式的核心对比指标"""
    # 1. 持续学习平均准确率
    _, continual_acc_matrix = load_continual_forward_test()
    continual_avg_acc = np.diag(continual_acc_matrix.values).mean()  # 对角线（当前任务）平均准确率
    
    # 2. 独立学习平均准确率
    independent_final = pd.read_csv(os.path.join(INDEPENDENT_DIR, "independent_final_results.csv"))
    independent_avg_acc = independent_final["recent_accuracy"].mean()
    
    # 3. 联合学习平均准确率（修复路径拼接）
    JOINT_DIR = os.path.join(os.path.dirname(__file__), "0", "joint_results")
    
    joint_files = [f for f in os.listdir(JOINT_DIR) if re.match(r"joint_model_per_task_accuracy_\d+_tasks\.csv", f)]
    if not joint_files:
        raise FileNotFoundError(f"在 {JOINT_DIR} 下未找到 joint_model_per_task_accuracy_*_tasks.csv 文件")
    joint_file = sorted(joint_files)[-1]  # 取最新的文件
    print(f"自动匹配到联合训练文件：{joint_file}")



    joint_per_task_path = os.path.join(JOINT_DIR, joint_file)
    if not os.path.exists(joint_per_task_path):
        raise FileNotFoundError(f"联合训练 per-task 文件不存在：{joint_per_task_path}")
    joint_per_task = pd.read_csv(joint_per_task_path)
    joint_avg_acc = joint_per_task["accuracy"].mean()
    
    # 4. NC1均值
    nc_data = load_nc_time_series()
    continual_nc1 = nc_data["continual_final"]["nc1"].mean()
    independent_nc1 = nc_data["independent_final"]["nc1"].mean()
    joint_nc1 = joint_per_task["nc1"].mean()
    
    # 5. 死亡神经元平均比例
    dead_data = load_dead_neurons()
    continual_dead_avg = dead_data["continual_dead_ratio"].mean(axis=1).mean()
    independent_dead_avg = dead_data["independent_dead_ratio"].mean(axis=1).mean()
    
    # 6. 平均收敛轮数（简化：用迭代数近似）
    continual_iter = nc_data["continual_inter"]["iter"].max() / len(nc_data["continual_final"])
    independent_iter = nc_data["independent_inter"]["iter"].max() / len(nc_data["independent_final"])
    joint_iter = nc_data["joint_inter"]["total_trained_samples"].max() / 60000  # 按样本数折算
    
    # 构建对比表格
    compare_df = pd.DataFrame({
        "Training_Paradigm": ["Continual", "Independent", "Joint"],
        "Average_Accuracy": [continual_avg_acc, independent_avg_acc, joint_avg_acc],
        "Average_NC1": [continual_nc1, independent_nc1, joint_nc1],
        "Avg_Dead_Neuron_Ratio": [continual_dead_avg, independent_dead_avg, 0],  # 联合训练无死亡神经元统计
        "Average_Converge_Iter": [continual_iter, independent_iter, joint_iter]
    })
    
    return compare_df

def main():
    """主函数：生成所有预处理数据表格并保存"""
    # 创建输出目录
    output_dir = "preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 持续学习前向测试准确率矩阵
    _, acc_matrix_df = load_continual_forward_test()
    acc_matrix_df.to_csv(os.path.join(output_dir, "continual_acc_matrix.csv"), index=True)
    
    # 2. NC指标时间序列
    nc_data = load_nc_time_series()
    for key, df in nc_data.items():
        df.to_csv(os.path.join(output_dir, f"{key}_nc.csv"), index=False)
    
    # 3. 死亡神经元比例
    dead_data = load_dead_neurons()
    dead_data["continual_dead_ratio"].to_csv(os.path.join(output_dir, "continual_dead_ratio.csv"), index=False)
    dead_data["independent_dead_ratio"].to_csv(os.path.join(output_dir, "independent_dead_ratio.csv"), index=False)
    
    # 4. 终止条件统计
    continual_stats, independent_stats = load_termination_stats()
    termination_df = pd.DataFrame({
        "Paradigm": ["Continual", "Independent"],
        "Reached_Threshold": [continual_stats["reached_threshold"], independent_stats["reached_threshold"]],
        "Epoch_Exhausted": [continual_stats["epoch_exhausted"], independent_stats["epoch_exhausted"]],
        "Total_Tasks": [continual_stats["total_tasks"], independent_stats["total_tasks"]]
    })
    termination_df.to_csv(os.path.join(output_dir, "termination_stats.csv"), index=False)
    
    # 5. 三种范式对比表
    compare_df = load_paradigm_comparison()
    compare_df.to_csv(os.path.join(output_dir, "paradigm_comparison.csv"), index=False)
    
    print("✅ 数据预处理完成！所有表格已保存至 preprocessed_data/ 目录")

if __name__ == "__main__":
    main()
import os
import sys
import json
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.nets.linear import MyLinear
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from lop.nets.deep_ffnn import DeepFFNN
from lop.utils.miscellaneous import nll_accuracy, compute_matrix_rank_summaries
from lop.utils.neural_collapse import NC

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Optimize CUDA memory allocation (reduce fragmentation)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

plt.switch_backend('Agg')  

dead_neuron_threshold = 1e-8    

dead_neuron_threshold = 1e-8    

# ====================== 新增：通用死亡神经元计算函数 ======================
def calculate_dead_neurons(learner, x_task, num_hidden_layers, dev):
    """通用函数：计算当前模型的死亡神经元数量（训练后调用）"""
    learner.net.eval()  # 评估模式，避免BatchNorm等影响
    with torch.no_grad():
        m = learner.net.predict(x_task[:20000])[1]  # 用前20000样本，和原逻辑一致
        dead_neurons_list = []
        for rep_layer_idx in range(num_hidden_layers):
            neuron_activation_sums = m[rep_layer_idx].abs().sum(dim=0)
            dead = (neuron_activation_sums < dead_neuron_threshold).sum()
            dead_neurons_list.append(int(dead.item()))
        del m
    learner.net.train()  # 切回训练模式
    return dead_neurons_list
# ====================== 新增结束 ======================


class TrajectoryMap:
    def __init__(self, model, step=1, save_dir="figures"):
        self.model = model
        self.step = step
        self.save_dir = save_dir
        self.trajectory = []  
        os.makedirs(self.save_dir, exist_ok=True) 

    def record_trajectory(self, current_step, is_train=True):
        if not is_train or (current_step % self.step != 0):
            return
        
        with torch.no_grad():
            params_list = []
            for param in self.model.parameters():
                flat_param = param.data.cpu().flatten()
                params_list.append(flat_param)
                del flat_param
        
        if not params_list:
            return
        
        with torch.no_grad():
            flat_params = torch.cat(params_list)
            param_norm = flat_params.norm(2)  
            
            eps = 1e-12 
            if param_norm < eps:
                norm_params = flat_params / (param_norm + eps)
            else:
                norm_params = flat_params / param_norm  
            
            del flat_params, param_norm
        
        assert torch.isclose(norm_params.norm(2), torch.tensor(1.0), atol=1e-4), \
            f"Parameter normalization failed! Vector length = {norm_params.norm(2)}, which should be close to 1 as required by the paper"
        
        self.trajectory.append(norm_params)
        del norm_params

    def compute_trajectory_metrics(self):
        if len(self.trajectory) < 2:
            return {
                "traj_avg_cos": 0.0,    
                "traj_min_cos": 0.0,    
                "traj_max_cos": 0.0,   
                "traj_length": len(self.trajectory)  
            }
        
        with torch.no_grad():
            traj_tensor = torch.stack(self.trajectory)
            cos_sim_matrix = traj_tensor @ traj_tensor.T  
            cos_sim_matrix = torch.clamp(cos_sim_matrix, min=-1.0, max=1.0)
            upper_triangle = cos_sim_matrix.triu(diagonal=1)  
            valid_cos_vals = upper_triangle[upper_triangle != 0]  
            
            del traj_tensor, cos_sim_matrix, upper_triangle
        
        return {
            "traj_avg_cos": float(valid_cos_vals.mean().item()),  
            "traj_min_cos": float(valid_cos_vals.min().item()),  
            "traj_max_cos": float(valid_cos_vals.max().item()),  
            "traj_length": len(self.trajectory)                  
        }

    def plot_trajectory_map(self, task_idx, total_iterations):
        if len(self.trajectory) == 0:
            print(f"[Task {task_idx}] No parameter trajectory recorded, skipping plotting")
            return
        
        traj_tensor = torch.stack(self.trajectory)
        
        cos_sim_matrix = traj_tensor @ traj_tensor.T
        
        cos_sim_matrix = torch.clamp(cos_sim_matrix, min=-1.0, max=1.0)
        
        plt.figure(figsize=(10, 8))
       
        colors = [(1, 1, 0), (0, 1, 0), (0, 0, 1)]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("blu_green_yellow", colors, N=256)
        
        im = plt.imshow(cos_sim_matrix.numpy(), cmap=custom_cmap, interpolation="nearest")
        plt.colorbar(im, label="Cosine Similarity ")  
        plt.title(f"Task {task_idx} Trajectory Map (Total Iterations: {total_iterations})")
        
        total_records = len(traj_tensor)
        if total_records > 1:
            tick_positions = np.linspace(0, total_records - 1, min(21, total_records + 1), dtype=int)
            tick_labels = [f"{int(total_iterations * pos / (total_records - 1))}" for pos in tick_positions]
        else:
            tick_positions = [0]
            tick_labels = [f"{0}"]
        
        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.yticks(tick_positions, tick_labels)
        plt.tight_layout()  
        
        save_path = os.path.join(self.save_dir, f"task_{task_idx}_trajectory_map.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Task {task_idx}] Trajectory Map saved to: {save_path}")


def save_model_params_to_csv(model, save_dir, num_tasks):
    layer_groups = [
        {
            "layer_keys": ["in_layer.fc.weight", "in_layer.fc.bias"],
            "filename": f'joint_model_params_{num_tasks}_tasks_layer_0_input.csv'
        },
        {
            "layer_keys": ["layers.0.weight", "layers.0.bias"],
            "filename": f'joint_model_params_{num_tasks}_tasks_layer_1_hidden1.csv'
        },
        {
            "layer_keys": ["layers.2.weight", "layers.2.bias"],
            "filename": f'joint_model_params_{num_tasks}_tasks_layer_2_hidden2.csv'
        },
        {
            "layer_keys": ["layers.4.weight", "layers.4.bias"],
            "filename": f'joint_model_params_{num_tasks}_tasks_layer_3_hidden3.csv'
        },
        {
            "layer_keys": ["layers.6.weight", "layers.6.bias"],
            "filename": f'joint_model_params_{num_tasks}_tasks_layer_4_output.csv'
        }
    ]
    
    file_handles = {}
    for group in layer_groups:
        csv_path = os.path.join(save_dir, group["filename"])
        fh = open(csv_path, 'w', encoding='utf-8')
        fh.write('layer_name,param_flat_index,param_value,num_tasks\n')
        file_handles[group["filename"]] = fh
    
    for layer_name, param in model.named_parameters():
        target_group = None
        for group in layer_groups:
            if layer_name in group["layer_keys"]:
                target_group = group
                break
        if not target_group:
            continue  
        
        flat_param = param.data.cpu().flatten()
        fh = file_handles[target_group["filename"]]
        for idx, val in enumerate(flat_param):
            fh.write(f"{layer_name},{idx},{val.item():.10f},{num_tasks}\n")
    
    for fh in file_handles.values():
        fh.close()
    
    print(f"[Joint Training] Model parameters split into 5 CSV files in: {save_dir}")
    for group in layer_groups:
        csv_path = os.path.join(save_dir, group["filename"])
        print(f"  - {csv_path}")


def generate_task_samples(x_original, y_original, pixel_perm, data_perm, examples_per_task, dev):
    x_task = x_original[:, pixel_perm].clone()
    x_task, y_task = x_task[data_perm], y_original[data_perm].clone()
    
    x_task = x_task[:examples_per_task].to(dev)
    y_task = y_task[:examples_per_task].to(dev)
    return x_task, y_task


def release_tensor_memory(*tensors): # 优化内存小手段
    for tensor in tensors:
        if tensor is not None:
            del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ====================== 新增：前向测试函数 ======================
def forward_test(learner, tasks_permutations, x_original, y_original, current_task_idx, save_dir, dev):
    """
    前向测试：在第current_task_idx个任务训练完成后，测试当前网络对所有历史任务(T1~Tcurrent_task_idx)的性能
    """
    learner.net.eval()
    # 前向测试结果保存文件
    forward_test_file = os.path.join(save_dir, 'continual_forward_test_results.csv')
    
    # 初始化文件头（仅第一次创建时）
    if not os.path.exists(forward_test_file):
        headers = [
            'current_train_task_idx',  # 当前训练完成的任务ID
            'test_task_idx',           # 被测试的历史任务ID
            'accuracy',                # 该历史任务的测试准确率
            'nc1', 'nc2', 'nc3', 'nc4',# 该历史任务的神经坍缩指标
            'num_samples'              # 该任务的测试样本数
        ]
        with open(forward_test_file, 'w', encoding='utf-8') as f:
            f.write(','.join(headers) + '\n')
    
    examples_per_task = 60000
    print(f"\n=== Forward Test: Evaluate current model on all historical tasks (T1~T{current_task_idx}) ===")
    
    # 遍历所有历史任务（0 ~ current_task_idx）
    for test_task_idx in range(current_task_idx + 1):
        print(f"  Testing historical task {test_task_idx}...")
        # 获取该历史任务的像素/数据排列
        pixel_perm, data_perm = tasks_permutations[test_task_idx]
        # 生成该任务的测试样本
        x_task, y_task = generate_task_samples(
            x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
        )
        task_dataset = TensorDataset(x_task, y_task)
        task_dataloader = DataLoader(task_dataset, batch_size=1000)
        
        # 计算准确率
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for val_x, val_y in task_dataloader:
                val_output = learner.net(val_x)
                preds = torch.argmax(val_output, dim=1)
                total_correct += (preds == val_y).sum().item()
                total_samples += val_y.size(0)
        accuracy = total_correct / total_samples
        
        # 计算NC指标
        nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=task_dataloader, num_classes=10)
        #learner.net.train()  # 切回训练模式
        
        # 打印结果
        print(f"    Task {test_task_idx} - Accuracy: {accuracy:.6f}, NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
        
        # 写入CSV文件
        with open(forward_test_file, 'a', encoding='utf-8') as f:
            f.write(
                f"{current_task_idx},"          # 当前训练完成的任务ID
                f"{test_task_idx},"             # 被测试的历史任务ID
                f"{accuracy:.6f},"              # 准确率
                f"{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},"  # NC指标
                f"{total_samples}\n"            # 样本数
            )
        
        # 释放内存
        release_tensor_memory(x_task, y_task)
    
    learner.net.train()
    print(f"=== Forward Test completed! Results saved to {forward_test_file} ===")
# ====================== 新增结束 ======================

def test_joint_model_per_task(joint_learner, tasks_permutations, x_original, y_original, save_dir, num_tasks, dev):
    joint_learner.net.eval()
    
    per_task_acc_file = os.path.join(save_dir, f'joint_model_per_task_accuracy_{num_tasks}_tasks.csv')
    
    with open(per_task_acc_file, 'w', encoding='utf-8') as f:
        f.write('task_idx,accuracy,num_samples,num_tasks,nc1,nc2,nc3,nc4\n')
    
    examples_per_task = 60000
    print(f"\n[Joint Model Per-Task Test] Start testing {num_tasks} tasks (each with 60000 samples)...")
    for task_idx, (pixel_perm, data_perm) in enumerate(tasks_permutations):
        print(f"  Testing Task {task_idx}...")
        
        x_task, y_task = generate_task_samples(
            x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
        )
        
        with torch.no_grad():
            output = joint_learner.net(x_task)
            preds = torch.argmax(output, dim=1)
            
            total_correct = (preds == y_task).sum().item()
            total_samples = y_task.size(0)
            accuracy = total_correct / total_samples
            
            task_dataset = TensorDataset(x_task, y_task)
            task_dataloader = DataLoader(task_dataset, batch_size=1000)  
            nc1, nc2, nc3, nc4 = NC(model=joint_learner.net, data_loader=task_dataloader, num_classes=10)
            #learner.net.train()  # 切回训练模式
        
        print(f"  Task {task_idx} Accuracy: {accuracy:.6f} (Correct: {total_correct}/{total_samples})")
        print(f"  Task {task_idx} NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
        
        with open(per_task_acc_file, 'a', encoding='utf-8') as f:
            f.write(f"{task_idx},{accuracy:.6f},{total_samples},{num_tasks},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f}\n")
        
        release_tensor_memory(x_task, y_task)
    
    print(f"[Joint Model Per-Task Test] Results saved to: {per_task_acc_file}")
    
    joint_learner.net.train()

def create_model(params, input_size, classes_per_task, num_hidden_layers, num_features, dev):
    if params['agent'] == 'linear':
        net = MyLinear(
            input_size=input_size, num_outputs=classes_per_task
        )
        net.layers_to_log = []
    else:
        net = DeepFFNN(input_size=input_size, num_features=num_features, 
                      num_outputs=classes_per_task, num_hidden_layers=num_hidden_layers)
    
    if params['agent'] in ['bp', 'linear', "l2"]:
        learner = Backprop(
            net=net,
            step_size=params['step_size'],
            opt=params['opt'],
            loss='nll',
            weight_decay=params.get('weight_decay', 0),
            device=dev,
            to_perturb=params.get('to_perturb', False),
            perturb_scale=params.get('perturb_scale', 0.1),
        )
    elif params['agent'] in ['cbp']:
        learner = ContinualBackprop(
            net=net,
            step_size=params['step_size'],
            opt=params['opt'],
            loss='nll',
            maturity_threshold=params.get('mt', 100),
            decay_rate=params.get('decay_rate', 0.99),
            util_type=params.get('util_type', 'adaptable_contribution'),
            accumulate=True,
            device=dev,
        )
    return learner

def train_single_task(learner, x_original, y_original, task_idx, params, save_dir, traj_save_dir, 
                     num_hidden_layers, input_size, examples_per_task, change_after, 
                     mini_batch_size, rank_measure_period, iter, accuracies, 
                     weight_mag_sum, effective_ranks, approximate_ranks, 
                     approximate_ranks_abs, ranks, dead_neurons, dev):
    traj_recorder = TrajectoryMap(
        model=learner.net,
        step=params.get('traj_step', 100),
        save_dir=traj_save_dir
    )
    
    new_iter_start = iter
    
    pixel_permutation = np.random.permutation(input_size)
    data_permutation = np.random.permutation(len(y_original))  
    
    x_task, y_task = generate_task_samples(
        x_original, y_original, pixel_permutation, data_permutation, examples_per_task, dev
    )
    dataset = TensorDataset(x_task, y_task)
    dataloader = DataLoader(dataset, batch_size=1000)
    
    if params['agent'] != 'linear':
        with torch.no_grad():
            new_idx = int(iter / rank_measure_period)
            m = learner.net.predict(x_task[:20000])[1]
            task_start_approx_ranks = []
            task_start_dead_neurons = []
            for rep_layer_idx in range(num_hidden_layers):
                ranks[new_idx][rep_layer_idx], effective_ranks[new_idx][rep_layer_idx], \
                approx_rank_val, approximate_ranks_abs[new_idx][rep_layer_idx] = \
                    compute_matrix_rank_summaries(m=m[rep_layer_idx], use_scipy=True)
                task_start_approx_ranks.append(round(float(approx_rank_val.item()), 6))
                neuron_activation_sums = m[rep_layer_idx].abs().sum(dim=0)
                dead = (neuron_activation_sums < dead_neuron_threshold).sum()
                task_start_dead_neurons.append(int(dead.item()))
            #print(f'[Task {task_idx}] Initial approximate ranks: {task_start_approx_ranks}, Initial dead neurons: {task_start_dead_neurons}')
            del m
    
    intermediate_file = os.path.join(save_dir, '0_0.csv')
    if not os.path.exists(intermediate_file):
        intermediate_headers = [
            'task_idx', 'iter', 'nc1', 'nc2', 'nc3', 'nc4', 'full_accuracy',
            'traj_avg_cos', 'traj_min_cos', 'traj_max_cos', 'traj_length'
        ]
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            f.write(','.join(intermediate_headers) + '\n')
    
    #num_epochs = 10
    num_epochs = 10
    task_reached_threshold = False  
    total_train_steps = change_after * num_epochs
    nc1_interval = 10000  
    
    for epoch in range(num_epochs):
        if task_reached_threshold:
            break  
        print(f"\n[Task {task_idx}] Epoch {epoch+1}/{num_epochs}")
        
        epoch_permutation = np.random.permutation(examples_per_task)
        x_epoch = x_task[epoch_permutation]
        y_epoch = y_task[epoch_permutation]
        
        for start_idx in tqdm(range(0, change_after, mini_batch_size), desc=f"Task {task_idx} Epoch {epoch+1} Training"):
            start_idx = start_idx % examples_per_task
            batch_x = x_epoch[start_idx: start_idx + mini_batch_size]
            batch_y = y_epoch[start_idx: start_idx + mini_batch_size]
            
            loss, network_output = learner.learn(x=batch_x, target=batch_y)
            
            if params.get('to_log', False) and params['agent'] != 'linear':
                for idx, layer_idx in enumerate(learner.net.layers_to_log):
                    weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
            
            with torch.no_grad():
                accuracies[iter] = nll_accuracy(softmax(network_output, dim=1), batch_y).cpu()
            
            traj_recorder.record_trajectory(current_step=iter, is_train=True)
            
            if (iter - new_iter_start + 1) % nc1_interval == 0:
                nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=dataloader, num_classes=10)
                #learner.net.train()  # 切回训练模式
                
                total_correct = 0
                total_samples = 0
                with torch.no_grad():
                    for val_x, val_y in dataloader:
                        val_output = learner.net(val_x)
                        preds = torch.argmax(val_output, dim=1)
                        total_correct += (preds == val_y).sum().item()
                        total_samples += val_y.size(0)
                full_accuracy = total_correct / total_samples
                #0.96
                if full_accuracy >= 0.96:  
                    task_reached_threshold = True
                    print(f"[Task {task_idx}] Reached target accuracy {full_accuracy:.4f}, stopping early")
                
                current_traj_metrics = traj_recorder.compute_trajectory_metrics()
                
                print(f"\n[Task {task_idx} Epoch {epoch+1} Iter {iter}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
                print(f"[Task {task_idx} Epoch {epoch+1} Iter {iter}] Full Accuracy: {full_accuracy:.4f}")
                print(f"[Task {task_idx} Epoch {epoch+1} Iter {iter}] Traj Metrics - Avg Cos: {current_traj_metrics['traj_avg_cos']:.6f}, Min Cos: {current_traj_metrics['traj_min_cos']:.6f}, Max Cos: {current_traj_metrics['traj_max_cos']:.6f}, Length: {current_traj_metrics['traj_length']}")
                
                with open(intermediate_file, 'a', encoding='utf-8') as f:
                    f.write(
                        f"{task_idx},{iter},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},{full_accuracy:.6f},"
                        f"{current_traj_metrics['traj_avg_cos']:.6f},{current_traj_metrics['traj_min_cos']:.6f},"
                        f"{current_traj_metrics['traj_max_cos']:.6f},{current_traj_metrics['traj_length']:.0f}\n"
                    )
            
            del batch_x, batch_y
            iter += 1  
            
            if iter >= len(accuracies) or task_reached_threshold:
                break
    
    task_traj_metrics = traj_recorder.compute_trajectory_metrics()
    task_total_iters = iter - new_iter_start
    traj_recorder.plot_trajectory_map(task_idx=task_idx, total_iterations=task_total_iters)
    
    print("[DEBUG] After NC computation: model.training =", learner.net.training)
    nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=dataloader, num_classes=10)
   # learner.net.train()  # 切回训练模式
    print("[DEBUG] After net.train(): model.training =", learner.net.training)
    print(f"[Task {task_idx} End] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
    
    if params['agent'] != 'linear':
        if len(task_start_approx_ranks) != num_hidden_layers or len(task_start_dead_neurons) != num_hidden_layers:
            task_approx_ranks = [0.0] * num_hidden_layers
            task_dead_neurons = calculate_dead_neurons(learner, x_task, num_hidden_layers, dev)
        else:
            task_approx_ranks = task_start_approx_ranks
            task_dead_neurons = calculate_dead_neurons(learner, x_task, num_hidden_layers, dev)
    else:
        task_approx_ranks = [0.0] * num_hidden_layers
        task_dead_neurons = [0] * num_hidden_layers
    
    recent_acc = float(accuracies[new_iter_start:iter - 1].mean().item())
    
    task_end_file = os.path.join(save_dir, '0.csv')
    if not os.path.exists(task_end_file):
        approx_rank_headers = [f"approx_rank_{i+1}" for i in range(num_hidden_layers)]
        dead_neurons_headers = [f"dead_neurons_{i+1}" for i in range(num_hidden_layers)]
        all_headers = [
            'task_idx', 'nc1', 'nc2', 'nc3', 'nc4', 'recent_accuracy', 'reached_threshold'
        ] + approx_rank_headers + dead_neurons_headers + [
            'traj_avg_cos', 'traj_min_cos', 'traj_max_cos', 'traj_length'
        ]
        with open(task_end_file, 'w', encoding='utf-8') as f:
            f.write(','.join(all_headers) + '\n')
    
    with open(task_end_file, 'a', encoding='utf-8') as f:
        data_row = [
            str(task_idx),
            f"{nc1:.6f}", f"{nc2:.6f}", f"{nc3:.6f}", f"{nc4:.6f}",
            f"{recent_acc:.6f}", str(task_reached_threshold)
        ] + [f"{ar}" for ar in task_approx_ranks] + [f"{dn}" for dn in task_dead_neurons] + [
            f"{task_traj_metrics['traj_avg_cos']:.6f}",
            f"{task_traj_metrics['traj_min_cos']:.6f}",
            f"{task_traj_metrics['traj_max_cos']:.6f}",
            f"{task_traj_metrics['traj_length']:.0f}"
        ]
        f.write(','.join(data_row) + '\n')
    
    release_tensor_memory(x_task, y_task, x_epoch, y_epoch)
    return iter, learner, pixel_permutation, data_permutation, task_reached_threshold


def train_joint_model(params, tasks_permutations, x_original, y_original, save_dir, dev):
    n_tasks = len(tasks_permutations)
    samples_per_task = 60000  
    total_samples_per_epoch = n_tasks * samples_per_task  
    mini_batch_size = params.get('mini_batch_size', 1)
    #num_epochs = 10  0
    num_epochs = 10
    record_interval = total_samples_per_epoch // 6
    record_points = [i * record_interval for i in range(1, 7)]  
    
    joint_intermediate_file = os.path.join(save_dir, 'joint_intermediate_results.csv')
    if not os.path.exists(joint_intermediate_file):
        intermediate_headers = [
            'epoch', 'record_idx_in_epoch', 'total_trained_samples', 
            'accuracy', 'nc1', 'nc2', 'nc3', 'nc4', 'num_tasks'
        ]
        with open(joint_intermediate_file, 'w', encoding='utf-8') as f:
            f.write(','.join(intermediate_headers) + '\n')

    joint_x_list = []
    joint_y_list = []
    examples_per_task = 60000
    for pixel_perm, data_perm in tasks_permutations:
        x_task, y_task = generate_task_samples(
            x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
        )
        joint_x_list.append(x_task)
        joint_y_list.append(y_task)
    joint_x = torch.cat(joint_x_list)
    joint_y = torch.cat(joint_y_list)
    
    release_tensor_memory(*joint_x_list, *joint_y_list) # 释放中间变量内存
    
    full_dataset = TensorDataset(joint_x, joint_y)
    full_dataloader = DataLoader(full_dataset, batch_size=1000)  # 计算NC值使用
    
    input_size = joint_x.shape[1]
    classes_per_task = 10
    num_hidden_layers = params.get('num_hidden_layers', 1)
    num_features = params.get('num_features', 2000)
    learner = create_model(params, input_size, classes_per_task, num_hidden_layers, num_features, dev)
    
    joint_reached_threshold = False
    epoch_iter = 0  
    total_trained_samples_global = 0  
    
    for epoch in range(num_epochs):
        if joint_reached_threshold:
            break
        
        epoch_perm = np.random.permutation(total_samples_per_epoch)
        x_epoch = joint_x[epoch_perm]
        y_epoch = joint_y[epoch_perm]
        
        print(f"\nJoint Training Epoch {epoch+1}/{num_epochs}")
        current_total_samples = 0  
        recorded_points = set()  
        
        pbar = tqdm(
            range(0, total_samples_per_epoch, mini_batch_size),
            desc=f"Joint Training Epoch {epoch+1} Training"
        )
        
        for start_idx in pbar:
            batch_x = x_epoch[start_idx: start_idx + mini_batch_size]
            batch_y = y_epoch[start_idx: start_idx + mini_batch_size]
            if len(batch_x) == 0:
                continue
            
            learner.learn(x=batch_x, target=batch_y)
            epoch_iter += 1
            current_total_samples += len(batch_x)  
            total_trained_samples_global += len(batch_x)  
            
            for point in record_points:
                if point not in recorded_points and current_total_samples >= point:
                    nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=full_dataloader, num_classes=10)
                    #learner.net.train()  # 切回训练模式
                    
                    total_correct = 0
                    total_samples_eval = 0
                    with torch.no_grad():
                        for val_x, val_y in full_dataloader:
                            val_output = learner.net(val_x)
                            preds = torch.argmax(val_output, dim=1)
                            total_correct += (preds == val_y).sum().item()
                            total_samples_eval += val_y.size(0)
                    full_accuracy = total_correct / total_samples_eval
                    
                    print(f"\n[Joint Training Epoch {epoch+1} Iter {epoch_iter}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
                    print(f"[Joint Training Epoch {epoch+1} Iter {epoch_iter}] Full Accuracy: {full_accuracy:.4f}")
                    
                    record_idx_in_epoch = len(recorded_points) + 1
                    
                    with open(joint_intermediate_file, 'a', encoding='utf-8') as f:
                        f.write(
                            f"{epoch+1},{record_idx_in_epoch},{total_trained_samples_global},"
                            f"{full_accuracy:.6f},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},{n_tasks}\n"
                        )
                    #0.96
                    if full_accuracy >= 0.96:
                        joint_reached_threshold = True
                        print(f"Joint Training Reached target accuracy {full_accuracy:.4f}, stopping early")
                        pbar.close()  
                        break
                    
                    recorded_points.add(point)  
            if joint_reached_threshold:
                break
            
            del batch_x, batch_y
        
        if joint_reached_threshold:
            break
    
    nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=full_dataloader, num_classes=10)
    #learner.net.train()  # 切回训练模式
    total_correct = 0
    total_samples_eval = 0
    with torch.no_grad():
        for val_x, val_y in full_dataloader:
            val_output = learner.net(val_x)
            preds = torch.argmax(val_output, dim=1)
            total_correct += (preds == val_y).sum().item()
            total_samples_eval += val_y.size(0)
    final_accuracy = total_correct / total_samples_eval
    print(f"NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
    
    release_tensor_memory(joint_x, joint_y, x_epoch, y_epoch)
    
    return learner, final_accuracy, nc1, nc2, nc3, nc4, joint_reached_threshold

def train_independent_tasks(params, tasks_permutations, x_original, y_original, save_dir, traj_save_dir, 
                           num_hidden_layers, input_size, examples_per_task, change_after, 
                           mini_batch_size, dev):
    independent_model_dir = os.path.join(save_dir, 'independent_models')
    os.makedirs(independent_model_dir, exist_ok=True)
    
    independent_intermediate_file = os.path.join(save_dir, 'independent_intermediate_results.csv')
    if not os.path.exists(independent_intermediate_file):
        intermediate_headers = [
            'task_idx', 'iter', 'nc1', 'nc2', 'nc3', 'nc4', 'full_accuracy',
            'traj_avg_cos', 'traj_min_cos', 'traj_max_cos', 'traj_length'
        ]
        with open(independent_intermediate_file, 'w', encoding='utf-8') as f:
            f.write(','.join(intermediate_headers) + '\n')
    
    independent_final_file = os.path.join(save_dir, 'independent_final_results.csv')
    if not os.path.exists(independent_final_file):
        approx_rank_headers = [f"approx_rank_{i+1}" for i in range(num_hidden_layers)]
        dead_neurons_headers = [f"dead_neurons_{i+1}" for i in range(num_hidden_layers)]
        all_headers = [
            'task_idx', 'nc1', 'nc2', 'nc3', 'nc4', 'recent_accuracy', 'reached_threshold'
        ] + approx_rank_headers + dead_neurons_headers + [
            'traj_avg_cos', 'traj_min_cos', 'traj_max_cos', 'traj_length'
        ]
        with open(independent_final_file, 'w', encoding='utf-8') as f:
            f.write(','.join(all_headers) + '\n')
    
    num_tasks = len(tasks_permutations)
    print(f"\n=== Starting Independent Training for {num_tasks} Tasks ===")
    
    for task_idx in range(num_tasks):
        print(f"\n--- Independent Training for Task {task_idx} ---")
        
        current_learner = create_model(params, input_size, 10, num_hidden_layers, params.get('num_features', 2000), dev)
        
        total_examples = int((task_idx + 1) * change_after * 10)
        total_iters = int(total_examples / mini_batch_size)
        rank_measure_period = 60000
        accuracies = torch.zeros(total_iters, dtype=torch.float)
        weight_mag_sum = torch.zeros((total_iters, num_hidden_layers + 1), dtype=torch.float)
        effective_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        approximate_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        approximate_ranks_abs = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        dead_neurons = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        
        iter_count = 0
        pixel_perm, data_perm = tasks_permutations[task_idx]
        
        traj_recorder = TrajectoryMap(
            model=current_learner.net,
            step=params.get('traj_step', 100),
            save_dir=os.path.join(traj_save_dir, 'independent')
        )
        os.makedirs(os.path.join(traj_save_dir, 'independent'), exist_ok=True)
        
        new_iter_start = iter_count
        
        x_task, y_task = generate_task_samples(
            x_original, y_original, pixel_perm, data_perm, examples_per_task, dev
        )
        dataset = TensorDataset(x_task, y_task)
        dataloader = DataLoader(dataset, batch_size=1000)
        
        task_start_approx_ranks = []
        task_start_dead_neurons = []
        if params['agent'] != 'linear':
            with torch.no_grad():
                new_idx = int(iter_count / rank_measure_period)
                m = current_learner.net.predict(x_task[:20000])[1]
                for rep_layer_idx in range(num_hidden_layers):
                    ranks[new_idx][rep_layer_idx], effective_ranks[new_idx][rep_layer_idx], \
                    approx_rank_val, approximate_ranks_abs[new_idx][rep_layer_idx] = \
                        compute_matrix_rank_summaries(m=m[rep_layer_idx], use_scipy=True)
                    task_start_approx_ranks.append(round(float(approx_rank_val.item()), 6))
                    neuron_activation_sums = m[rep_layer_idx].abs().sum(dim=0)
                    dead = (neuron_activation_sums < dead_neuron_threshold).sum()
                    task_start_dead_neurons.append(int(dead.item()))
                #print(f'[Independent Task {task_idx}] Initial approximate ranks: {task_start_approx_ranks}, Initial dead neurons: {task_start_dead_neurons}')
                del m
        
        #num_epochs = 10
        num_epochs = 10
        task_reached_threshold = False
        total_train_steps = change_after * num_epochs
        nc1_interval = 10000
        
        for epoch in range(num_epochs):
            if task_reached_threshold:
                break
            print(f"\n[Independent Task {task_idx}] Epoch {epoch+1}/{num_epochs}")
            
            epoch_permutation = np.random.permutation(examples_per_task)
            x_epoch = x_task[epoch_permutation]
            y_epoch = y_task[epoch_permutation]
            
            for start_idx in tqdm(range(0, change_after, mini_batch_size), desc=f"Independent Task {task_idx} Epoch {epoch+1} Training"):
                start_idx = start_idx % examples_per_task
                batch_x = x_epoch[start_idx: start_idx + mini_batch_size]
                batch_y = y_epoch[start_idx: start_idx + mini_batch_size]
                
                loss, network_output = current_learner.learn(x=batch_x, target=batch_y)
                
                if params.get('to_log', False) and params['agent'] != 'linear':
                    for idx, layer_idx in enumerate(current_learner.net.layers_to_log):
                        weight_mag_sum[iter_count][idx] = current_learner.net.layers[layer_idx].weight.data.abs().sum()
                
                with torch.no_grad():
                    accuracies[iter_count] = nll_accuracy(softmax(network_output, dim=1), batch_y).cpu()
                
                traj_recorder.record_trajectory(current_step=iter_count, is_train=True)
                
                if (iter_count - new_iter_start + 1) % nc1_interval == 0:
                    nc1, nc2, nc3, nc4 = NC(model=current_learner.net, data_loader=dataloader, num_classes=10)
                    #learner.net.train()  # 切回训练模式
                    
                    total_correct = 0
                    total_samples = 0
                    with torch.no_grad():
                        for val_x, val_y in dataloader:
                            val_output = current_learner.net(val_x)
                            preds = torch.argmax(val_output, dim=1)
                            total_correct += (preds == val_y).sum().item()
                            total_samples += val_y.size(0)
                    full_accuracy = total_correct / total_samples
                    #0.96
                    if full_accuracy >= 0.96:
                        task_reached_threshold = True
                        print(f"[Independent Task {task_idx}] Reached target accuracy {full_accuracy:.4f}, stopping early")
                    
                    current_traj_metrics = traj_recorder.compute_trajectory_metrics()
                    
                    print(f"\n[Independent Task {task_idx} Epoch {epoch+1} Iter {iter_count}] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
                    print(f"[Independent Task {task_idx} Epoch {epoch+1} Iter {iter_count}] Full Accuracy: {full_accuracy:.4f}")
                    print(f"[Independent Task {task_idx} Epoch {epoch+1} Iter {iter_count}] Traj Metrics - Avg Cos: {current_traj_metrics['traj_avg_cos']:.6f}, Min Cos: {current_traj_metrics['traj_min_cos']:.6f}, Max Cos: {current_traj_metrics['traj_max_cos']:.6f}, Length: {current_traj_metrics['traj_length']}")
                    
                    with open(independent_intermediate_file, 'a', encoding='utf-8') as f:
                        f.write(
                            f"{task_idx},{iter_count},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},{full_accuracy:.6f},"
                            f"{current_traj_metrics['traj_avg_cos']:.6f},{current_traj_metrics['traj_min_cos']:.6f},"
                            f"{current_traj_metrics['traj_max_cos']:.6f},{current_traj_metrics['traj_length']:.0f}\n"
                        )
                
                del batch_x, batch_y
                iter_count += 1
                
                if iter_count >= len(accuracies) or task_reached_threshold:
                    break
        
        task_traj_metrics = traj_recorder.compute_trajectory_metrics()
        task_total_iters = iter_count - new_iter_start
        traj_recorder.plot_trajectory_map(task_idx=task_idx, total_iterations=task_total_iters)
        
        current_learner.net.eval()
        nc1, nc2, nc3, nc4 = NC(model=current_learner.net, data_loader=dataloader, num_classes=10)
        #learner.net.train()  # 切回训练模式
        current_learner.net.train()
        print(f"[Independent Task {task_idx} End] NC1: {nc1:.6f}, NC2: {nc2:.6f}, NC3: {nc3:.6f}, NC4: {nc4:.6f}")
        
        if params['agent'] != 'linear':
            if len(task_start_approx_ranks) != num_hidden_layers or len(task_start_dead_neurons) != num_hidden_layers:
                task_approx_ranks = [0.0] * num_hidden_layers
                task_dead_neurons = calculate_dead_neurons(current_learner, x_task, num_hidden_layers, dev)
            else:
                task_approx_ranks = task_start_approx_ranks
                task_dead_neurons = calculate_dead_neurons(current_learner, x_task, num_hidden_layers, dev)
        else:
            task_approx_ranks = [0.0] * num_hidden_layers
            task_dead_neurons = [0] * num_hidden_layers
        
        recent_acc = float(accuracies[new_iter_start:iter_count - 1].mean().item())
        
        with open(independent_final_file, 'a', encoding='utf-8') as f:
            data_row = [
                str(task_idx),
                f"{nc1:.6f}", f"{nc2:.6f}", f"{nc3:.6f}", f"{nc4:.6f}",
                f"{recent_acc:.6f}", str(task_reached_threshold)
            ] + [f"{ar}" for ar in task_approx_ranks] + [f"{dn}" for dn in task_dead_neurons] + [
                f"{task_traj_metrics['traj_avg_cos']:.6f}",
                f"{task_traj_metrics['traj_min_cos']:.6f}",
                f"{task_traj_metrics['traj_max_cos']:.6f}",
                f"{task_traj_metrics['traj_length']:.0f}"
            ]
            f.write(','.join(data_row) + '\n')
        
        model_path = os.path.join(independent_model_dir, f'independent_model_task_{task_idx}.pth')
        torch.save(current_learner.net.state_dict(), model_path)
        print(f"[Independent Task {task_idx}] Model saved to: {model_path}")
        
        release_tensor_memory(x_task, y_task, x_epoch, y_epoch)
    
    print(f"\n=== Independent Training Completed for All {num_tasks} Tasks ===")
    print(f"Intermediate results saved to: {independent_intermediate_file}")
    print(f"Final results saved to: {independent_final_file}")

def online_expr(params: dict):
    agent_type = params['agent']
    initial_num_tasks = 0  # 初始任务数量设为0，后续根据num_examples和change_after动态计算任务数量
    max_task_increase = 600  # 最大任务数量增加限制，防止过多任务导致训练时间过长
    target_accuracy = 0.96#0.96  
    
    num_tasks = params.get('num_tasks', 200)
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"] / params["change_after"])
    
    save_dir = '3/'  
    os.makedirs(save_dir, exist_ok=True)  
    
    # Delete 0.csv file if exists
    zero_csv_path = os.path.join(save_dir, '0.csv')
    # if os.path.exists(zero_csv_path):
    #     os.remove(zero_csv_path)
    #     print(f"Deleted existing 0.csv file at: {zero_csv_path}")
    
    model_save_dir = os.path.join(save_dir, 'models')
    joint_results_dir = os.path.join(save_dir, 'joint_results')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(joint_results_dir, exist_ok=True)
    
    step_size = params['step_size']             
    opt = params['opt']                         
    weight_decay = params.get('weight_decay', 0)                            
    use_gpu = params.get('use_gpu', 0)                                 
    dev = 'cpu'                                 
    if use_gpu == 1:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if dev == torch.device("cuda"):    
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    to_log = params.get('to_log', False)                              
    num_features = params.get('num_features', 2000)                         
    change_after = params.get('change_after', 60000)                          
    to_perturb = params.get('to_perturb', False)                          
    perturb_scale = params.get('perturb_scale', 0.1)                         
    num_hidden_layers = params.get('num_hidden_layers', 1)                       
    
    mini_batch_size = params.get('mini_batch_size', 1)                         
    decay_rate = params.get('decay_rate', 0.99)                           
    maturity_threshold = params.get('mt', 100)                    
    util_type = params.get('util_type', 'adaptable_contribution')        
    
    traj_step = params.get('traj_step', 100)  
    traj_save_dir = os.path.join(save_dir, 'trajectory_maps')  
    os.makedirs(traj_save_dir, exist_ok=True) 
    
    nc_data_dir = params['data_dir'].rstrip('/') + '_NC/'  
    os.makedirs(nc_data_dir, exist_ok=True)
    
    classes_per_task = 10
    images_per_class = 6000
    input_size = 784
    examples_per_task = images_per_class * classes_per_task
    
    with open('data/mnist_', 'rb') as f:
        x_original, y_original, _, _ = pickle.load(f)
        if use_gpu == 1:
            x_original = x_original.to(dev)
            y_original = y_original.to(dev)
    
    task_params = []
    tasks_permutations = []  
    all_joint_results = []
    
    joint_results_file = os.path.join(joint_results_dir, 'joint_training_results.csv')
    with open(joint_results_file, 'w', encoding='utf-8') as f:
        f.write('num_tasks,accuracy,nc1,nc2,nc3,nc4,capacity_reached,joint_reached_threshold\n')
    
    max_possible_tasks = min(num_tasks, initial_num_tasks + max_task_increase)  
    
    current_learner = None
    iter_count = 0
    network_capacity = 0  
    task_reached_threshold_list = []
    
    # # Step 1: Complete continuous training for all tasks
    for task_idx in range(max_possible_tasks):
        print(f"\n=== Starting Task {task_idx} (Current Total Tasks: {len(task_params)+1}) ===")
        
        total_examples = int((len(task_params)+1) * change_after * 10)  
        total_iters = int(total_examples / mini_batch_size)
        rank_measure_period = 60000
        accuracies = torch.zeros(total_iters, dtype=torch.float)
        weight_mag_sum = torch.zeros((total_iters, num_hidden_layers + 1), dtype=torch.float)
        effective_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        approximate_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        approximate_ranks_abs = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        dead_neurons = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
        
        if current_learner is None:
            current_learner = create_model(params, input_size, classes_per_task, 
                                          num_hidden_layers, num_features, dev)
        
        print(f"\n--- Continuous Training for Task {task_idx} ---")
        iter_count, current_learner, pixel_perm, data_perm, task_reached_threshold = train_single_task(
            current_learner, x_original, y_original, task_idx, params, save_dir, traj_save_dir,
            num_hidden_layers, input_size, examples_per_task, change_after,
            mini_batch_size, rank_measure_period, iter_count, accuracies,
            weight_mag_sum, effective_ranks, approximate_ranks,
            approximate_ranks_abs, ranks, dead_neurons, dev
        )
        
        model_path = os.path.join(model_save_dir, f'model_task_{task_idx}.pth')
        torch.save(current_learner.net.state_dict(), model_path)
        task_params.append(model_path)
        # pixel_perm = np.random.permutation(input_size)
        # data_perm = np.random.permutation(len(y_original))  
        tasks_permutations.append((pixel_perm, data_perm))  # 这里做的不是很好
        task_reached_threshold_list.append(task_reached_threshold)
        
        # ====================== 新增：调用前向测试 ======================
        # 第task_idx个任务训练完成后，测试所有历史任务(T1~Ttask_idx)
        forward_test(
            learner=current_learner,
            tasks_permutations=tasks_permutations,
            x_original=x_original,
            y_original=y_original,
            current_task_idx=task_idx,
            save_dir=save_dir,
            dev=dev
        )
        # ====================== 新增结束 ======================
    
    # Step 2: Independent training after continuous training
    train_independent_tasks(
        params=params,
        tasks_permutations=tasks_permutations,
        x_original=x_original,
        y_original=y_original,
        save_dir=save_dir,
        traj_save_dir=traj_save_dir,
        num_hidden_layers=num_hidden_layers,
        input_size=input_size,
        examples_per_task=examples_per_task,
        change_after=change_after,
        mini_batch_size=mini_batch_size,
        dev=dev
    )
    
    # Step 3: Joint training after independent training (original joint training logic)
    print(f"\n--- Start Joint Training After Independent Training ---")
    joint_learner, joint_acc, nc1, nc2, nc3, nc4, joint_reached_threshold = train_joint_model(
        params, tasks_permutations, x_original, y_original, joint_results_dir, dev
    )
    
    joint_model_path = os.path.join(joint_results_dir, f'joint_model_{len(tasks_permutations)}_tasks.pth')
    torch.save(joint_learner.net.state_dict(), joint_model_path)
    
    save_model_params_to_csv(joint_learner.net, joint_results_dir, len(tasks_permutations))
    
    test_joint_model_per_task(joint_learner, tasks_permutations, x_original, y_original, joint_results_dir, len(tasks_permutations), dev)
    
    capacity_reached = not joint_reached_threshold
    all_joint_results.append({
        'num_tasks': len(tasks_permutations),
        'accuracy': joint_acc,
        'nc1': nc1, 'nc2': nc2, 'nc3': nc3, 'nc4': nc4,
        'capacity_reached': capacity_reached,
        'joint_reached_threshold': joint_reached_threshold
    })
    
    with open(joint_results_file, 'a', encoding='utf-8') as f:
        f.write(f"{len(tasks_permutations)},{joint_acc:.6f},{nc1:.6f},{nc2:.6f},{nc3:.6f},{nc4:.6f},"
                f"{capacity_reached},{joint_reached_threshold}\n")
    
    print(f"[Joint Training Results] Tasks: {len(tasks_permutations)}, Accuracy: {joint_acc:.4f}, "
          f"Capacity Reached: {capacity_reached}")
    
    if capacity_reached:
        network_capacity = len(tasks_permutations) - 1  
        print(f"\nNetwork Capacity Reached! Maximum learnable tasks n = {network_capacity}")
    else:
        print(f"Joint Training Success")
    
    if network_capacity == 0 and len(task_params) > 0:
        network_capacity = len(task_params)
        print(f"\nAll tasks completed, Network Capacity n = {network_capacity}")
    
    capacity_file = os.path.join(save_dir, 'network_capacity.txt')
    with open(capacity_file, 'w', encoding='utf-8') as f:
        f.write(f"Maximum learnable tasks (n): {network_capacity}\n")
        f.write(f"Target accuracy threshold: {target_accuracy}\n")
        f.write(f"Training epochs limit: 10\n")
    
    print(f"\nExperiment Complete! Network Capacity: {network_capacity}")

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path to the config file for the experiment",
                        type=str, default='temp_cfg/1.json')
    parser.add_argument('--change_after', type=int, default=None,
                        help="(optional) override change_after from config")
    args = parser.parse_args(arguments)
    cfg_file = args.c
    with open(cfg_file, 'r') as f:
        params = json.load(f)
    if args.change_after is not None:
        params['change_after'] = int(args.change_after)
    online_expr(params)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
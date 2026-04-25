import os
import random
from random import shuffle
import sys
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd  # 用于数据整理
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.nets.linear import MyLinear
from torch.nn.functional import softmax
from lop.nets.deep_ffnn import DeepFFNN
from lop.utils.miscellaneous import nll_accuracy, compute_matrix_rank_summaries
from TrajectoryMap import TrajectoryMap
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from lop.utils.neural_collapse import NC

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "training_ckpt"
TASK_CURRENT_SUFFIX = "_current.pth"
TASK_FINAL_SUFFIX = "_final.pth"

seed = 42

SHEET_NAMES = ["训练精度", "测试精度", "NC1", "NC2", "NC3", "NC4"]

def save_task_current_checkpoint(
    task_idx, current_epoch, iter_num, learner, params,
    accuracies, weight_mag_sum, ranks, effective_ranks,
    approximate_ranks, approximate_ranks_abs, dead_neurons
):
    # 创建检查点目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 构造当前任务的临时检查点文件名和路径（同名文件会覆盖，保证仅保留最新轮数）
    current_task_ckpt_name = f"{CHECKPOINT_PREFIX}_task{task_idx}{TASK_CURRENT_SUFFIX}"
    current_task_ckpt_path = os.path.join(CHECKPOINT_DIR, current_task_ckpt_name)
    
    # 收集需要保存的数据（重点保留current_epoch，记录已训练轮数）
    checkpoint_data = {
        # 训练进度信息
        "task_idx": task_idx,
        "current_epoch": current_epoch,  # 关键：当前任务已完成的epoch（中断后从该epoch继续）
        "iter": iter_num,
        "num_tasks": params.get("num_tasks", 20),
        "total_epochs_per_task": params.get("epochs_per_task", 10),  # 记录每个任务的总轮数
        
        # 模型参数
        "model_state_dict": learner.net.state_dict(),
        
        # 学习器额外状态
        "learner_state": getattr(learner, "state_dict", lambda: {})(),
        
        # 随机数状态
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        
        # 日志张量数据
        "accuracies": accuracies,
        "weight_mag_sum": weight_mag_sum,
        "ranks": ranks,
        "effective_ranks": effective_ranks,
        "approximate_ranks": approximate_ranks,
        "approximate_ranks_abs": approximate_ranks_abs,
        "dead_neurons": dead_neurons,
        
        # 配置参数
        "params": params
    }
    
    # 保存当前任务的临时检查点（覆盖旧文件，仅保留最新轮数）
    torch.save(checkpoint_data, current_task_ckpt_path)
    print(f"\n当前任务（{task_idx}）临时检查点（epoch{current_epoch}）已保存（覆盖旧轮数）：{current_task_ckpt_path}")

def save_task_final_checkpoint(
    task_idx, total_epochs, iter_num, learner, params,
    accuracies, weight_mag_sum, ranks, effective_ranks,
    approximate_ranks, approximate_ranks_abs, dead_neurons
):
    # 创建检查点目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 构造当前任务的最终检查点文件名和路径（永久保留，不覆盖其他任务）
    current_task_final_ckpt_name = f"{CHECKPOINT_PREFIX}_task{task_idx}{TASK_FINAL_SUFFIX}"
    current_task_final_ckpt_path = os.path.join(CHECKPOINT_DIR, current_task_final_ckpt_name)
    
    # 收集需要保存的数据（标记任务已完成，保留完整状态）
    checkpoint_data = {
        # 训练进度信息
        "task_idx": task_idx,
        "current_epoch": total_epochs - 1,  # 任务完成，最后一轮epoch索引
        "iter": iter_num,
        "num_tasks": params.get("num_tasks", 20),
        "task_completed": True,  # 标记任务已完成
        
        # 模型参数
        "model_state_dict": learner.net.state_dict(),
        
        # 学习器额外状态
        "learner_state": getattr(learner, "state_dict", lambda: {})(),
        
        # 随机数状态
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        
        # 日志张量数据
        "accuracies": accuracies,
        "weight_mag_sum": weight_mag_sum,
        "ranks": ranks,
        "effective_ranks": effective_ranks,
        "approximate_ranks": approximate_ranks,
        "approximate_ranks_abs": approximate_ranks_abs,
        "dead_neurons": dead_neurons,
        
        # 配置参数
        "params": params
    }
    
    # 保存任务最终检查点（永久保留，不覆盖）
    torch.save(checkpoint_data, current_task_final_ckpt_path)
    print(f"\n当前任务（{task_idx}）最终检查点已永久保存：{current_task_final_ckpt_path}")
    
    # 清理该任务的临时检查点（任务已完成，临时文件冗余，可选保留）
    current_task_ckpt_name = f"{CHECKPOINT_PREFIX}_task{task_idx}{TASK_CURRENT_SUFFIX}"
    current_task_ckpt_path = os.path.join(CHECKPOINT_DIR, current_task_ckpt_name)
    if os.path.exists(current_task_ckpt_path):
        os.remove(current_task_ckpt_path)
        print(f"已清理任务（{task_idx}）的临时检查点（冗余文件）")

def find_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    # 查找所有未完成任务的临时检查点（taskX_current.pth）
    task_current_ckpt_files = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith(CHECKPOINT_PREFIX) and f.endswith(TASK_CURRENT_SUFFIX)
    ]
    
    if task_current_ckpt_files:
        # 提取临时检查点的任务索引，找到最新的未完成任务（仅可能有一个，因为同名覆盖）
        task_ckpt_info = []
        for filename in task_current_ckpt_files:
            try:
                task_str = filename.split("task")[1].split(TASK_CURRENT_SUFFIX)[0]
                task_idx = int(task_str)
                task_ckpt_info.append((task_idx, filename))
            except (IndexError, ValueError):
                continue
        
        if task_ckpt_info:
            # 按任务索引降序排序，返回最新的未完成任务临时检查点
            task_ckpt_info.sort(key=lambda x: x[0], reverse=True)
            latest_current_filename = task_ckpt_info[0][1]
            print(f"找到未完成任务的临时检查点：{latest_current_filename}（将恢复已训练轮数，避免重复）")
            return os.path.join(CHECKPOINT_DIR, latest_current_filename)
    
    # 若无线时检查点，查找已完成任务的最终检查点（taskX_final.pth）
    task_final_ckpt_files = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith(CHECKPOINT_PREFIX) and f.endswith(TASK_FINAL_SUFFIX)
    ]
    
    if not task_final_ckpt_files:
        return None
    
    # 提取最终检查点的任务索引，找到最新的已完成任务
    task_ckpt_info = []
    for filename in task_final_ckpt_files:
        try:
            task_str = filename.split("task")[1].split(TASK_FINAL_SUFFIX)[0]
            task_idx = int(task_str)
            task_ckpt_info.append((task_idx, filename))
        except (IndexError, ValueError):
            continue
    
    if not task_ckpt_info:
        return None
    
    # 按任务索引降序排序，返回最新的已完成任务最终检查点
    task_ckpt_info.sort(key=lambda x: x[0], reverse=True)
    latest_final_filename = task_ckpt_info[0][1]
    print(f"找到已完成任务的最终检查点：{latest_final_filename}（将从下一个任务开始训练）")
    return os.path.join(CHECKPOINT_DIR, latest_final_filename)

def load_checkpoint(ckpt_path, learner, params):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点文件不存在：{ckpt_path}")
    
    # 载入检查点数据
    checkpoint_data = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    # 配置一致性校验
    saved_params = checkpoint_data["params"]
    if saved_params.get("agent") != params.get("agent") or \
       saved_params.get("step_size") != params.get("step_size"):
        print("警告：当前配置与检查点配置存在差异，可能导致续训/复现结果异常！")
        print(f"检查点配置：agent={saved_params.get('agent')}, step_size={saved_params.get('step_size')}")
        print(f"当前配置：agent={params.get('agent')}, step_size={params.get('step_size')}")
    
    # 模型参数
    learner.net.load_state_dict(checkpoint_data["model_state_dict"])
    print("模型参数已恢复")
    
    # 学习器状态
    if hasattr(learner, "load_state_dict") and checkpoint_data["learner_state"]:
        learner.load_state_dict(checkpoint_data["learner_state"])
        print("学习器状态已恢复")
    
    # 随机数状态
    torch.set_rng_state(checkpoint_data["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint_data["torch_cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(checkpoint_data["torch_cuda_rng_state"])
    np.random.set_state(checkpoint_data["numpy_rng_state"])
    random.setstate(checkpoint_data["python_rng_state"])
    print("随机数状态已恢复")
    
    # 恢复训练进度
    task_idx = checkpoint_data["task_idx"]
    current_epoch = checkpoint_data.get("current_epoch", 0)  # 已完成的epoch（下一轮从该值开始）
    iter_num = checkpoint_data["iter"]
    task_completed = checkpoint_data.get("task_completed", False)  # 标记任务是否已完成
    
    if task_completed:
        # 任务已完成：下一轮训练从下一个任务开始，当前任务无需再训练
        print(f"训练进度已恢复：任务{task_idx}（已完成，总轮数{current_epoch+1}），迭代{iter_num}")
        task_idx += 1  # 跳转到下一个任务
        start_epoch = 0  # 下一个任务从epoch0开始
    else:
        # 任务未完成：从中断的epoch继续训练（current_epoch是已完成的轮数，下一轮从该值开始）
        print(f"训练进度已恢复：任务{task_idx}（未完成，已训练到epoch{current_epoch}），迭代{iter_num}")
        start_epoch = current_epoch  # 直接从已训练的epoch继续，避免重复
    
    # 恢复日志张量（保证实验复现可追溯完整训练日志）
    accuracies = checkpoint_data["accuracies"]
    weight_mag_sum = checkpoint_data["weight_mag_sum"]
    ranks = checkpoint_data["ranks"]
    effective_ranks = checkpoint_data["effective_ranks"]
    approximate_ranks = checkpoint_data["approximate_ranks"]
    approximate_ranks_abs = checkpoint_data["approximate_ranks_abs"]
    dead_neurons = checkpoint_data["dead_neurons"]
    print("日志张量数据已恢复")
    
    return (task_idx, start_epoch, iter_num, accuracies, weight_mag_sum,
            ranks, effective_ranks, approximate_ranks,
            approximate_ranks_abs, dead_neurons)

def online_expr(params: {}): # 持续训练
    # 定义Excel保存路径
    excel_save_path = "1.xlsx"
    excel_dir = os.path.dirname(excel_save_path)
    if excel_dir and not os.path.exists(excel_dir):
        os.makedirs(excel_dir, exist_ok=True)

    agent_type = params['agent']
    num_tasks = 200
    if 'num_tasks' in params.keys():
        num_tasks = params['num_tasks']
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"] / params["change_after"])
    num_tasks = 800  # 测试时缩小后的任务数

    # 原训练超参数（保持不变）
    step_size = params['step_size']
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev = 'cpu'
    to_log = False
    num_features = 2000
    change_after = 10 * 6000
    to_perturb = False
    perturb_scale = 0.1
    num_hidden_layers = 1
    mini_batch_size = 1
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'adaptable_contribution'

    if 'to_log' in params.keys():
        to_log = params['to_log']
    if 'weight_decay' in params.keys():
        weight_decay = params['weight_decay']
    if 'num_features' in params.keys():
        num_features = params['num_features']
    if 'change_after' in params.keys():
        change_after = params['change_after']
    if 'use_gpu' in params.keys():
        if params['use_gpu'] == 1:
            use_gpu = 1
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if dev == torch.device("cuda"):
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'to_perturb' in params.keys():
        to_perturb = params['to_perturb']
    if 'perturb_scale' in params.keys():
        perturb_scale = params['perturb_scale']
    if 'num_hidden_layers' in params.keys():
        num_hidden_layers = params['num_hidden_layers']
    if 'mini_batch_size' in params.keys():
        mini_batch_size = params['mini_batch_size']
    if 'replacement_rate' in params.keys():
        replacement_rate = params['replacement_rate']
    if 'decay_rate' in params.keys():
        decay_rate = params['decay_rate']
    if 'maturity_threshold' in params.keys():
        maturity_threshold = params['mt']
    if 'util_type' in params.keys():
        util_type = params['util_type']

    classes_per_task = 10
    images_per_class = 6000
    input_size = 784
    net = DeepFFNN(input_size=input_size, num_features=num_features, num_outputs=classes_per_task,
                   num_hidden_layers=num_hidden_layers)

    if agent_type == 'linear':
        net = MyLinear(
            input_size=input_size, num_outputs=classes_per_task
        )
        net.layers_to_log = []

    if agent_type in ['bp', 'linear', "l2"]:
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            weight_decay=weight_decay,
            device=dev,
            to_perturb=to_perturb,
            perturb_scale=perturb_scale,
        )
    elif agent_type in ['cbp']:
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            decay_rate=decay_rate,
            util_type=util_type,
            accumulate=True,
            device=dev,
        )

    accuracy = nll_accuracy
    examples_per_task = images_per_class * classes_per_task
    epochs = 10  # 每个任务的总轮数
    total_examples = int(num_tasks * change_after * epochs)
    total_iters = int(total_examples / mini_batch_size)
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks / 10)

    ckpt_path = find_latest_checkpoint()
    resume_training = ckpt_path is not None
    
    rank_measure_period = 60000

    if resume_training:
        # 载入最新检查点，恢复状态（重点恢复start_epoch，避免重复训练轮数）
        (start_task_idx, start_epoch, iter, accuracies, weight_mag_sum,
         ranks, effective_ranks, approximate_ranks,
         approximate_ranks_abs, dead_neurons) = load_checkpoint(ckpt_path, learner, params)
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 从头训练，初始化日志张量和训练进度
        start_task_idx = 0
        start_epoch = 0
        iter = 0
        accuracies = torch.zeros(total_iters, dtype=torch.float, device='cpu')
        weight_mag_sum = torch.zeros((total_iters, num_hidden_layers + 1), dtype=torch.float, device='cpu')
        
        effective_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float, device='cpu')
        approximate_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float, device='cpu')
        approximate_ranks_abs = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float, device='cpu')
        ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float, device='cpu')
        dead_neurons = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float, device='cpu')
    
    is_checked = False

    # 数据集加载
    with open('data/mnist_', 'rb+') as f:
        x_train, y_train, x_test, y_test = pickle.load(f)
        if use_gpu == 1:
            x_train = x_train.to(dev)
            y_train = y_train.to(dev)
            x_test = x_test.to(dev)
            y_test = y_test.to(dev)

    permutation = load_data(file='data/permutation')
    task_permutations = permutation['task_permutations']

    for task_idx in range(start_task_idx, num_tasks):
        print(f"\n========== 开始训练任务 {task_idx} ==========")
        total_epochs_per_task = epochs  # 每个任务的总轮数
        
        # 确定当前任务的起始epoch（从中断点恢复，避免重复）
        task_start_epoch = start_epoch + 1 if task_idx == start_task_idx else 0
        
        for epoch in range(task_start_epoch, total_epochs_per_task):
            print(f" Task {task_idx} - Epoch {epoch}:")
            new_iter_start = iter

            x, y = x_train[:, task_permutations[task_idx]], y_train
            data_permutation = np.random.permutation(examples_per_task)
            x, y = x[data_permutation], y[data_permutation]

            if agent_type != 'linear':
                with torch.no_grad():
                    new_idx = int(iter / rank_measure_period)
                    m = net.predict(x[:2000])[1]
                    for rep_layer_idx in range(num_hidden_layers):
                        ranks[new_idx][rep_layer_idx], effective_ranks[new_idx][rep_layer_idx], \
                            approximate_ranks[new_idx][rep_layer_idx], approximate_ranks_abs[new_idx][rep_layer_idx] = \
                            compute_matrix_rank_summaries(m=m[rep_layer_idx], use_scipy=True)
                        dead_neurons[new_idx][rep_layer_idx] = (m[rep_layer_idx].abs().sum(dim=0) == 0).sum()
                    print('approximate rank: ', approximate_ranks[new_idx], ', dead neurons: ', dead_neurons[new_idx])

            for start_idx in tqdm(range(0, change_after, mini_batch_size)):
                start_idx = start_idx % examples_per_task
                batch_x = x[start_idx: start_idx + mini_batch_size]
                batch_y = y[start_idx: start_idx + mini_batch_size]

                loss, network_output = learner.learn(x=batch_x, target=batch_y)

                if to_log and agent_type != 'linear':
                    for idx, layer_idx in enumerate(learner.net.layers_to_log):
                        weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
                with torch.no_grad():
                    accuracies[iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()
                iter += 1

                if iter % 10000 == 0:
                    current_iter_data = {
                        "训练精度": {"迭代次数": iter, "总体": None, "任务数据": []},
                        "测试精度": {"迭代次数": iter, "总体": None, "任务数据": []},
                        "NC1": {"迭代次数": iter, "总体": None, "任务数据": []},
                        "NC2": {"迭代次数": iter, "总体": None, "任务数据": []},
                        "NC3": {"迭代次数": iter, "总体": None, "任务数据": []},
                        "NC4": {"迭代次数": iter, "总体": None, "任务数据": []}
                    }

                    x_full, y_full = None, None
                    train_task_acc = []
                    nc1_task = []
                    nc2_task = []
                    nc3_task = []
                    nc4_task = []
                    for t_idx in range(task_idx+1):
                        with torch.no_grad():
                            x_train_permuted = x_train[:, task_permutations[t_idx]]
                            network_output, _ = learner.net.predict(x_train_permuted)
                            train_accuracy = accuracy(softmax(network_output, dim=1), y_train).cpu()

                            # NC计算
                            dataset = TensorDataset(x_train_permuted, y_train)
                            dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
                            nc1, nc2, nc3, nc4 = NC(model=learner.net, data_loader=dataloader, num_classes=10)

                            if t_idx == 0:
                                x_full, y_full = x_train_permuted, y_train
                            else:
                                x_full = torch.cat((x_full, x_train_permuted), dim=0)
                                y_full = torch.cat((y_full, y_train), dim=0)

                        # 收集单个任务数据
                        train_task_acc.append(train_accuracy.item())
                        nc1_task.append(nc1)
                        nc2_task.append(nc2)
                        nc3_task.append(nc3)
                        nc4_task.append(nc4)

                    # 计算训练均值和整体NC
                    train_acc_mean = np.mean(train_task_acc)
                    current_iter_data["训练精度"]["总体"] = train_acc_mean
                    current_iter_data["训练精度"]["任务数据"] = train_task_acc

                    with torch.no_grad():
                        concat_dataset = ConcatDataset([TensorDataset(x_train[:, task_permutations[t]], y_train) for t in range(task_idx+1)])
                        dataloader = DataLoader(concat_dataset, batch_size=256, shuffle=False)
                        overall_nc1, overall_nc2, overall_nc3, overall_nc4 = NC(model=learner.net, data_loader=dataloader, num_classes=10)

                    # 填充NC的总体和任务数据
                    for nc_sheet, overall_nc, task_nc in zip(
                        ["NC1", "NC2", "NC3", "NC4"],
                        [overall_nc1, overall_nc2, overall_nc3, overall_nc4],
                        [nc1_task, nc2_task, nc3_task, nc4_task]
                    ):
                        current_iter_data[nc_sheet]["总体"] = overall_nc
                        current_iter_data[nc_sheet]["任务数据"] = task_nc

                    test_task_acc = []
                    for t_idx in range(task_idx+1):
                        with torch.no_grad():
                            x_test_permuted = x_test[:, task_permutations[t_idx]]
                            network_output, _ = learner.net.predict(x_test_permuted)
                            test_accuracy = accuracy(softmax(network_output, dim=1), y_test).cpu()
                        test_task_acc.append(test_accuracy.item())

                    # 计算测试均值
                    test_acc_mean = np.mean(test_task_acc)
                    current_iter_data["测试精度"]["总体"] = test_acc_mean
                    current_iter_data["测试精度"]["任务数据"] = test_task_acc

                    write_to_multi_sheet_excel(current_iter_data, excel_save_path, task_idx+1)
            
            save_task_current_checkpoint(
                task_idx=task_idx,
                current_epoch=epoch,  # 记录当前已完成的epoch（下一轮从epoch+1开始）
                iter_num=iter,
                learner=learner,
                params=params,
                accuracies=accuracies,
                weight_mag_sum=weight_mag_sum,
                ranks=ranks,
                effective_ranks=effective_ranks,
                approximate_ranks=approximate_ranks,
                approximate_ranks_abs=approximate_ranks_abs,
                dead_neurons=dead_neurons
            )
            
            # 重置当前任务的起始epoch（仅当前任务生效，避免后续任务继承）
            if task_idx == start_task_idx:
                start_epoch = 0
        
        save_task_final_checkpoint(
            task_idx=task_idx,
            total_epochs=total_epochs_per_task,
            iter_num=iter,
            learner=learner,
            params=params,
            accuracies=accuracies,
            weight_mag_sum=weight_mag_sum,
            ranks=ranks,
            effective_ranks=effective_ranks,
            approximate_ranks=approximate_ranks,
            approximate_ranks_abs=approximate_ranks_abs,
            dead_neurons=dead_neurons
        )
        
        # 任务结束清理显存
        print(f"\n任务{task_idx}训练完成，近期精度均值：{accuracies[new_iter_start:iter - 1].mean():.4f}")
        torch.cuda.empty_cache()

        if task_idx % save_after_every_n_tasks == 0:
            data = {
                'accuracies': accuracies.cpu(),
                'weight_mag_sum': weight_mag_sum.cpu(),
                'ranks': ranks.cpu(),
                'effective_ranks': effective_ranks.cpu(),
                'approximate_ranks': approximate_ranks.cpu(),
                'abs_approximate_ranks': approximate_ranks_abs.cpu(),
                'dead_neurons': dead_neurons.cpu(),
            }

        data = {
            'accuracies': accuracies.cpu(),
            'weight_mag_sum': weight_mag_sum.cpu(),
            'ranks': ranks.cpu(),
            'effective_ranks': effective_ranks.cpu(),
            'approximate_ranks': approximate_ranks.cpu(),
            'abs_approximate_ranks': approximate_ranks_abs.cpu(),
            'dead_neurons': dead_neurons.cpu(),
        }

def write_to_multi_sheet_excel(current_iter_data, file_path, task_count):
    column_names = ["迭代次数", "总体"] + [f"任务{i}" for i in range(task_count)]

    sheet_dfs = {}
    for sheet_name in SHEET_NAMES:
        # 提取当前sheet的单行走数据
        iter_num = current_iter_data[sheet_name]["迭代次数"]
        overall_val = current_iter_data[sheet_name]["总体"]
        task_data = current_iter_data[sheet_name]["任务数据"]

        # 补全数据长度（防止任务数据缺失，填充NaN）
        task_data_padded = task_data + [np.nan] * (task_count - len(task_data))
        single_row_data = [iter_num, overall_val] + task_data_padded

        # 转换为DataFrame
        df_new = pd.DataFrame([single_row_data], columns=column_names)
        sheet_dfs[sheet_name] = df_new

    try:
        if os.path.exists(file_path):
            # 文件存在：读取原有各sheet数据，拼接新数据后重新写入
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                for sheet_name in SHEET_NAMES:
                    # 读取原有sheet数据
                    df_existing = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
                    # 对齐列名（新任务列扩展，补充原有数据的空值）
                    for col in column_names:
                        if col not in df_existing.columns:
                            df_existing[col] = np.nan
                    # 拼接新数据（仅保留共同列，保证结构一致）
                    df_combined = pd.concat([df_existing[column_names], sheet_dfs[sheet_name]], ignore_index=True)
                    # 写入当前sheet（替换原有内容，保持其他sheet不变）
                    df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # 文件不存在：新建文件，写入所有sheet的初始数据
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name in SHEET_NAMES:
                    sheet_dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(f"写入多sheet Excel失败：{e}")

def save_data(file, data):
    parent_dir = os.path.dirname(file)
    os.makedirs(parent_dir, exist_ok=True)
    with open(file, 'wb+') as f:
        pickle.dump(data, f)

def load_data(file: str) -> dict:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path to the file containing the parameters for the experiment",
                        type=str, default='temp_cfg/0.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    online_expr(params)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
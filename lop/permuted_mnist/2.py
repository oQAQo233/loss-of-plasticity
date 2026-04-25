import os
import random
from random import shuffle
import sys
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
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

CHECKPOINT_DIR = "joint_checkpoints"
CHECKPOINT_PREFIX = "joint_training_ckpt"
TASK_CURRENT_SUFFIX = "_current.pth"
TASK_FINAL_SUFFIX = "_final.pth"

# 定义全局常量：6个sheet名称
SHEET_NAMES = ["训练精度", "测试精度", "NC1", "NC2", "NC3", "NC4"]

# seed = 42
# torch.manual_seed(seed)                # PyTorch CPU 种子
# torch.cuda.manual_seed(seed)           # PyTorch GPU 种子
# torch.cuda.manual_seed_all(seed)       # 多GPU场景
# np.random.seed(seed)                   # NumPy 种子
# random.seed(seed)                      # Python 原生随机种子

def reset_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_joint_current_checkpoint(
    joint_k, epoch, iter_num, joint_learner, params,
    accuracies, accuracies_train, accuracies_test, weight_mag_sum,
    ranks, effective_ranks, approximate_ranks, approximate_ranks_abs, dead_neurons
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 构造当前k的临时检查点路径
    current_ckpt_name = f"{CHECKPOINT_PREFIX}_k{joint_k}{TASK_CURRENT_SUFFIX}"
    current_ckpt_path = os.path.join(CHECKPOINT_DIR, current_ckpt_name)
    
    # 收集保存数据（含随机数状态，保证复现一致性）
    checkpoint_data = {
        "joint_k": joint_k,
        "epoch": epoch,
        "iter": iter_num,
        "params": params,
        "model_state_dict": joint_learner.net.state_dict(),
        "learner_state": getattr(joint_learner, "state_dict", lambda: {})(),
        # 随机数状态
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        # 日志数据
        "accuracies": accuracies,
        "accuracies_train": accuracies_train,
        "accuracies_test": accuracies_test,
        "weight_mag_sum": weight_mag_sum,
        "ranks": ranks,
        "effective_ranks": effective_ranks,
        "approximate_ranks": approximate_ranks,
        "approximate_ranks_abs": approximate_ranks_abs,
        "dead_neurons": dead_neurons
    }
    
    # 保存临时检查点（覆盖旧文件，仅保留当前k最新epoch状态）
    torch.save(checkpoint_data, current_ckpt_path)
    print(f"\n联合任务k={joint_k} 临时检查点（epoch{epoch}）已保存：{current_ckpt_path}")

def save_joint_final_checkpoint(
    joint_k, epoch, iter_num, joint_learner, params,
    accuracies, accuracies_train, accuracies_test, weight_mag_sum,
    ranks, effective_ranks, approximate_ranks, approximate_ranks_abs, dead_neurons
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 构造当前k的最终检查点路径（永久保留，不覆盖其他k）
    final_ckpt_name = f"{CHECKPOINT_PREFIX}_k{joint_k}{TASK_FINAL_SUFFIX}"
    final_ckpt_path = os.path.join(CHECKPOINT_DIR, final_ckpt_name)
    
    # 收集保存数据（标记任务完成，方便续训判断）
    checkpoint_data = {
        "joint_k": joint_k,
        "epoch": epoch,
        "iter": iter_num,
        "joint_task_completed": True,
        "params": params,
        "model_state_dict": joint_learner.net.state_dict(),
        "learner_state": getattr(joint_learner, "state_dict", lambda: {})(),
        # 随机数状态
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        # 日志数据
        "accuracies": accuracies,
        "accuracies_train": accuracies_train,
        "accuracies_test": accuracies_test,
        "weight_mag_sum": weight_mag_sum,
        "ranks": ranks,
        "effective_ranks": effective_ranks,
        "approximate_ranks": approximate_ranks,
        "approximate_ranks_abs": approximate_ranks_abs,
        "dead_neurons": dead_neurons
    }
    
    # 保存最终检查点（永久保留，不覆盖其他k）
    torch.save(checkpoint_data, final_ckpt_path)
    print(f"\n联合任务k={joint_k} 最终检查点已永久保存：{final_ckpt_path}")
    
    # 清理该k的临时检查点（冗余，可选保留，此处删除以节省空间）
    current_ckpt_name = f"{CHECKPOINT_PREFIX}_k{joint_k}{TASK_CURRENT_SUFFIX}"
    current_ckpt_path = os.path.join(CHECKPOINT_DIR, current_ckpt_name)
    if os.path.exists(current_ckpt_path):
        os.remove(current_ckpt_path)
        print(f"已清理联合任务k={joint_k} 临时检查点（冗余文件）")

def find_latest_joint_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    # 查找未完成k的临时检查点
    current_ckpt_files = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith(CHECKPOINT_PREFIX) and f.endswith(TASK_CURRENT_SUFFIX)
    ]
    if current_ckpt_files:
        ckpt_info = []
        for filename in current_ckpt_files:
            try:
                k_str = filename.split("_k")[1].split(TASK_CURRENT_SUFFIX)[0]
                joint_k = int(k_str)
                ckpt_info.append((joint_k, filename))
            except (IndexError, ValueError):
                continue
        if ckpt_info:
            ckpt_info.sort(key=lambda x: x[0], reverse=True)
            latest_filename = ckpt_info[0][1]
            print(f"找到未完成联合任务k={ckpt_info[0][0]} 的临时检查点：{latest_filename}")
            return os.path.join(CHECKPOINT_DIR, latest_filename)
    
    # 查找已完成k的最终检查点
    final_ckpt_files = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith(CHECKPOINT_PREFIX) and f.endswith(TASK_FINAL_SUFFIX)
    ]
    if not final_ckpt_files:
        return None
    
    ckpt_info = []
    for filename in final_ckpt_files:
        try:
            k_str = filename.split("_k")[1].split(TASK_FINAL_SUFFIX)[0]
            joint_k = int(k_str)
            ckpt_info.append((joint_k, filename))
        except (IndexError, ValueError):
            continue
    if not ckpt_info:
        return None
    
    ckpt_info.sort(key=lambda x: x[0], reverse=True)
    latest_filename = ckpt_info[0][1]
    print(f"找到已完成联合任务k={ckpt_info[0][0]} 的最终检查点：{latest_filename}")
    return os.path.join(CHECKPOINT_DIR, latest_filename)

def load_joint_checkpoint(ckpt_path, joint_learner, params):
    """
    载入联合任务k的检查点，恢复模型、epoch、iter、随机数状态
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点文件不存在：{ckpt_path}")
    
    checkpoint_data = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    # 配置一致性校验
    saved_params = checkpoint_data["params"]
    if saved_params.get("agent") != params.get("agent") or saved_params.get("step_size") != params.get("step_size"):
        print("警告：当前配置与检查点配置存在差异，可能影响复现结果！")
    
    # 恢复模型和学习器状态
    joint_learner.net.load_state_dict(checkpoint_data["model_state_dict"])
    if hasattr(joint_learner, "load_state_dict") and checkpoint_data["learner_state"]:
        joint_learner.load_state_dict(checkpoint_data["learner_state"])
    
    # 恢复随机数状态
    torch.set_rng_state(checkpoint_data["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint_data["torch_cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(checkpoint_data["torch_cuda_rng_state"])
    np.random.set_state(checkpoint_data["numpy_rng_state"])
    random.setstate(checkpoint_data["python_rng_state"])
    
    # 恢复训练进度
    joint_k = checkpoint_data["joint_k"]
    epoch = checkpoint_data["epoch"]
    iter_num = checkpoint_data["iter"]
    task_completed = checkpoint_data.get("joint_task_completed", False)
    
    # 恢复日志数据
    accuracies = checkpoint_data["accuracies"]
    accuracies_train = checkpoint_data["accuracies_train"]
    accuracies_test = checkpoint_data["accuracies_test"]
    weight_mag_sum = checkpoint_data["weight_mag_sum"]
    ranks = checkpoint_data["ranks"]
    effective_ranks = checkpoint_data["effective_ranks"]
    approximate_ranks = checkpoint_data["approximate_ranks"]
    approximate_ranks_abs = checkpoint_data["approximate_ranks_abs"]
    dead_neurons = checkpoint_data["dead_neurons"]
    
    print(f"检查点载入完成：k={joint_k}，已训练到epoch={epoch}，iter={iter_num}")
    return (joint_k, epoch, iter_num, task_completed,
            accuracies, accuracies_train, accuracies_test,
            weight_mag_sum, ranks, effective_ranks,
            approximate_ranks, approximate_ranks_abs, dead_neurons)

def joint_expr(params: {}):  # 联合训练
    # 定义Excel保存路径
    excel_save_path = "2.xlsx"
    excel_dir = os.path.dirname(excel_save_path)
    if excel_dir and not os.path.exists(excel_dir):
        os.makedirs(excel_dir, exist_ok=True)

    agent_type = params['agent'] # 训练方法（bp）

    num_tasks = 200
    if 'num_tasks' in params.keys():
        num_tasks = params['num_tasks']
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"] / params["change_after"])

    num_tasks = 50 # 测试时缩小后的任务数

    # 原训练超参数
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

    accuracy_threshold = 0.96

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
    input_size = 784  # 28x28
    num_features = num_features
    num_hidden_layers = num_hidden_layers

    accuracy = nll_accuracy
    examples_per_task = images_per_class * classes_per_task
    epochs = 10 # 新增
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks / 10)

    weight_mag_sum = torch.zeros((int(num_tasks * change_after * epochs), num_hidden_layers + 1), dtype=torch.float, device='cpu')

    rank_measure_period = 60000
    total_examples_max = int(num_tasks * change_after * epochs)
    effective_ranks = torch.zeros((int(total_examples_max / rank_measure_period), num_hidden_layers), dtype=torch.float, device='cpu')
    approximate_ranks = torch.zeros((int(total_examples_max / rank_measure_period), num_hidden_layers), dtype=torch.float, device='cpu')
    approximate_ranks_abs = torch.zeros((int(total_examples_max / rank_measure_period), num_hidden_layers), dtype=torch.float, device='cpu')
    ranks = torch.zeros((int(total_examples_max / rank_measure_period), num_hidden_layers), dtype=torch.float, device='cpu')
    dead_neurons = torch.zeros((int(total_examples_max / rank_measure_period), num_hidden_layers), dtype=torch.float, device='cpu')

    is_checked = False

    with open('data/mnist_', 'rb+') as f:
        x_train, y_train, x_test, y_test = pickle.load(f)
        if use_gpu == 1:
            x_train = x_train.to(dev)
            y_train = y_train.to(dev)
            x_test = x_test.to(dev)
            y_test = y_test.to(dev)

    permutation = load_data(file='data/permutation')

    task_permutations = permutation['task_permutations']  # 保存每个任务的像素排列
    x_permutation_indices = permutation['x_permutation_indices']  # 保存每个任务的数据-对应排列

    ckpt_path = find_latest_joint_checkpoint()
    start_k = 20
    if ckpt_path:
        # 先临时初始化一个模型和learner用于载入检查点（后续会覆盖）
        temp_net = DeepFFNN(input_size=input_size, num_features=num_features,
                            num_outputs=classes_per_task, num_hidden_layers=num_hidden_layers)
        temp_learner = Backprop(net=temp_net, step_size=step_size, opt=opt, loss='nll',
                                weight_decay=weight_decay, device=dev, to_perturb=to_perturb,
                                perturb_scale=perturb_scale)
        # 载入检查点获取最新k
        (latest_k, _, _, task_completed, _, _, _, _, _, _, _, _, _) = load_joint_checkpoint(ckpt_path, temp_learner, params)
        start_k = latest_k + 10 if task_completed else latest_k # 暂时间隔设为10
        print(f"即将从k={start_k}开始训练（已完成/未完成k={latest_k}）")

    # 不同数量任务联合训练
    for k in range(start_k, num_tasks + 1, 10):  # range(num_tasks, num_tasks + 1) k为任务数量
        print(f"\n--- Evaluating joint training with {k} tasks ---")

        reset_random_seeds()

        total_examples = int(k * change_after * epochs)
        total_iters = int(total_examples / mini_batch_size)

        # 日志张量移到CPU，避免GPU显存占用
        accuracies = torch.zeros(total_iters, dtype=torch.float, device='cpu')
        accuracies_train = torch.zeros(k, dtype=torch.float, device='cpu') # 联合训练验证精度
        accuracies_test = torch.zeros(k, dtype=torch.float, device='cpu') # 联合训练测试精度

        task_permutations_k = task_permutations[:k]
        x_permutation_indices_k = x_permutation_indices[:k * change_after]

        if agent_type == 'linear':
            joint_net = MyLinear(
                input_size=input_size, num_outputs=classes_per_task
            )
            joint_net.layers_to_log = []
        else:
            joint_net = DeepFFNN(
                input_size=input_size,
                num_features=num_features,
                num_outputs=classes_per_task,
                num_hidden_layers=num_hidden_layers
            )

        if agent_type in ['bp', 'linear', "l2"]:
            joint_learner = Backprop(
                net=joint_net,
                step_size=step_size,
                opt=opt,
                loss='nll',
                weight_decay=weight_decay,
                device=dev,
                to_perturb=to_perturb,
                perturb_scale=perturb_scale,
            )
        elif agent_type in ['cbp']:
            joint_learner = ContinualBackprop(
                net=joint_net,
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

        current_ckpt_name = f"{CHECKPOINT_PREFIX}_k{k}{TASK_CURRENT_SUFFIX}"
        current_ckpt_path = os.path.join(CHECKPOINT_DIR, current_ckpt_name)
        resume_k_training = os.path.exists(current_ckpt_path)
        start_epoch = 0
        iter = 0
        if resume_k_training:
            (_, latest_epoch, iter, _, accuracies, accuracies_train, accuracies_test,
             weight_mag_sum, ranks, effective_ranks, approximate_ranks,
             approximate_ranks_abs, dead_neurons) = load_joint_checkpoint(current_ckpt_path, joint_learner, params)
            print(f"恢复k={k}的训练，从epoch={start_epoch}、iter={iter}开始")

            start_epoch = latest_epoch + 1

        total_loss = 0
        num_batches = len(x_permutation_indices_k) // mini_batch_size

        early_stop_triggered = False # 早停

        for epoch in range(start_epoch, epochs):
            print(f" Epoch {epoch}:")
            shuffle(x_permutation_indices_k) # 打乱数据顺序

            overall_accuracies = []

            for batch_idx in tqdm(range(num_batches)):
                start = batch_idx * mini_batch_size
                end = start + mini_batch_size

                batch_indices = x_permutation_indices_k[start:end]
                batch_x = torch.zeros((mini_batch_size, input_size), device=dev)
                batch_y = torch.zeros((mini_batch_size,), dtype=torch.long, device=dev)
                for i in range(mini_batch_size):
                    original_idx, permutation_idx = batch_indices[i]
                    pixel_permutation = task_permutations_k[permutation_idx]
                    batch_x[i] = x_train[original_idx, pixel_permutation]
                    batch_y[i] = y_train[original_idx]

                # train the network
                loss, network_output = joint_learner.learn(x=batch_x, target=batch_y)

                # log accuracy
                with torch.no_grad():
                    batch_accuracy = accuracy(softmax(network_output, dim=1), batch_y).cpu()
                    if iter < len(accuracies):
                        accuracies[iter] = batch_accuracy
                iter += 1
                total_loss += loss.item()

                if iter % 10000 == 0 and iter <= len(accuracies):
                    current_iter_data = {
                        "训练精度": {"联合任务数": k, "迭代次数": iter, "总体": None, "任务数据": []},
                        "测试精度": {"联合任务数": k, "迭代次数": iter, "总体": None, "任务数据": []},
                        "NC1": {"联合任务数": k, "迭代次数": iter, "总体": None, "任务数据": []},
                        "NC2": {"联合任务数": k, "迭代次数": iter, "总体": None, "任务数据": []},
                        "NC3": {"联合任务数": k, "迭代次数": iter, "总体": None, "任务数据": []},
                        "NC4": {"联合任务数": k, "迭代次数": iter, "总体": None, "任务数据": []}
                    }

                    train_task_acc = []
                    nc1_task = []
                    nc2_task = []
                    nc3_task = []
                    nc4_task = []
                    for t_idx in range(k):
                        with torch.no_grad():
                            x_train_permuted = x_train[:, task_permutations[t_idx]]
                            network_output, _ = joint_learner.net.predict(x_train_permuted)
                            train_accuracy = accuracy(softmax(network_output, dim=1), y_train).cpu()

                            # 减小batch_size优化显存
                            dataset = TensorDataset(x_train_permuted, y_train)
                            dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
                            nc1, nc2, nc3, nc4 = NC(model=joint_learner.net, data_loader=dataloader, num_classes=10)

                        # 收集单个任务数据
                        train_task_acc.append(train_accuracy.item())
                        nc1_task.append(nc1)
                        nc2_task.append(nc2)
                        nc3_task.append(nc3)
                        nc4_task.append(nc4)
                        accuracies_train[t_idx] = train_accuracy

                    # 计算训练均值和整体NC
                    train_acc_mean = np.mean(train_task_acc)
                    current_iter_data["训练精度"]["总体"] = train_acc_mean
                    current_iter_data["训练精度"]["任务数据"] = train_task_acc

                    # 计算整体NC（优化：使用ConcatDataset避免大张量拼接）
                    with torch.no_grad():
                        concat_dataset = ConcatDataset([TensorDataset(x_train[:, task_permutations[t]], y_train) for t in range(k)])
                        dataloader = DataLoader(concat_dataset, batch_size=256, shuffle=False)
                        overall_nc1, overall_nc2, overall_nc3, overall_nc4 = NC(model=joint_learner.net, data_loader=dataloader, num_classes=10)

                    # 填充NC的总体和任务数据
                    for nc_sheet, overall_nc, task_nc in zip(
                        ["NC1", "NC2", "NC3", "NC4"],
                        [overall_nc1, overall_nc2, overall_nc3, overall_nc4],
                        [nc1_task, nc2_task, nc3_task, nc4_task]
                    ):
                        current_iter_data[nc_sheet]["总体"] = overall_nc
                        current_iter_data[nc_sheet]["任务数据"] = task_nc

                    # 3. 收集测试数据
                    test_task_acc = []
                    for t_idx in range(k):  # 测试不同序列
                        with torch.no_grad():
                            x_test_permuted = x_test[:, task_permutations[t_idx]]
                            network_output, _ = joint_learner.net.predict(x_test_permuted)
                            test_accuracy = accuracy(softmax(network_output, dim=1), y_test).cpu()

                        test_task_acc.append(test_accuracy.item())
                        accuracies_test[t_idx] = test_accuracy

                    # 计算测试均值
                    test_acc_mean = np.mean(test_task_acc)
                    current_iter_data["测试精度"]["总体"] = test_acc_mean
                    current_iter_data["测试精度"]["任务数据"] = test_task_acc

                    # 4. 写入多sheet Excel（新增联合任务数k）
                    write_to_multi_sheet_excel(current_iter_data, excel_save_path, k)

                    overall_accuracies.append(train_acc_mean)
                    overall_accuracies.append(test_acc_mean)

            if not all(acc > accuracy_threshold for acc in overall_accuracies):
                save_joint_current_checkpoint(
                    joint_k=k,
                    epoch=epoch,
                    iter_num=iter,
                    joint_learner=joint_learner,
                    params=params,
                    accuracies=accuracies,
                    accuracies_train=accuracies_train,
                    accuracies_test=accuracies_test,
                    weight_mag_sum=weight_mag_sum,
                    ranks=ranks,
                    effective_ranks=effective_ranks,
                    approximate_ranks=approximate_ranks,
                    approximate_ranks_abs=approximate_ranks_abs,
                    dead_neurons=dead_neurons
                )
            else:
                early_stop_triggered = True
                print(f"\n早停触发！k={k} 当前迭代总体精度均>0.96（训练：{train_acc_mean:.4f}，测试：{test_acc_mean:.4f}）")
                break

        final_epoch = epoch if early_stop_triggered else epochs - 1
        save_joint_final_checkpoint(
            joint_k=k, epoch=final_epoch, iter_num=iter, joint_learner=joint_learner, params=params,
            accuracies=accuracies, accuracies_train=accuracies_train, accuracies_test=accuracies_test,
            weight_mag_sum=weight_mag_sum, ranks=ranks, effective_ranks=effective_ranks,
            approximate_ranks=approximate_ranks, approximate_ranks_abs=approximate_ranks_abs,
            dead_neurons=dead_neurons
        )

        # 任务结束后清理显存，避免累积
        task_acc = accuracies[:min(iter, len(accuracies)) - 1].mean() if iter > 0 else 0.0
        print(f"  Task 0-{k-1} accuracy: {task_acc:.4f}")
        torch.cuda.empty_cache()

def write_to_multi_sheet_excel(current_iter_data, file_path, task_count):
    column_names = ["联合任务数", "迭代次数", "总体"] + [f"任务{i}" for i in range(task_count)]

    sheet_dfs = {}
    for sheet_name in SHEET_NAMES:
        # 提取当前sheet的单行走数据
        joint_task_num = current_iter_data[sheet_name]["联合任务数"]
        iter_num = current_iter_data[sheet_name]["迭代次数"]
        overall_val = current_iter_data[sheet_name]["总体"]
        task_data = current_iter_data[sheet_name]["任务数据"]

        # 补全数据长度（防止任务数据缺失，填充NaN）
        task_data_padded = task_data + [np.nan] * (task_count - len(task_data))
        single_row_data = [joint_task_num, iter_num, overall_val] + task_data_padded

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
    # 可能无对应文件，需要创建
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

    # online_expr(params)
    joint_expr(params)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
import os
import random
from random import shuffle
import sys
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.nets.linear import MyLinear
from torch.nn.functional import softmax
from lop.nets.deep_ffnn import DeepFFNN
from lop.utils.miscellaneous import nll_accuracy, compute_matrix_rank_summaries  # 计算矩阵秩的各种指标
from TrajectoryMap import TrajectoryMap

seed = 42
torch.manual_seed(seed)                # PyTorch CPU 种子
torch.cuda.manual_seed(seed)           # PyTorch GPU 种子
torch.cuda.manual_seed_all(seed)       # 多GPU场景
np.random.seed(seed)                   # NumPy 种子
random.seed(seed)                      # Python 原生随机种子

def online_expr(params: {}):  # 独立训练

    agent_type = params['agent'] # 训练方法（bp）

    num_tasks = 200
    if 'num_tasks' in params.keys():  # 顺序训练和联合训练共用同一参数（800）
        num_tasks = params['num_tasks']
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"] / params["change_after"])

    num_tasks = 100 # 测试时缩小后的任务数

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
            if dev == torch.device("cuda"):    torch.set_default_tensor_type('torch.cuda.FloatTensor')
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
    num_features = num_features # 2000（修改无法改进时间（可能因为并行计算））
    num_hidden_layers = num_hidden_layers
    
    accuracy = nll_accuracy
    examples_per_task = images_per_class * classes_per_task
    total_examples = int(num_tasks * change_after)
    total_iters = int(total_examples / mini_batch_size)
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks / 10)

    accuracies = torch.zeros(total_iters, dtype=torch.float)
    accuracies_train = torch.zeros(num_tasks, dtype=torch.float) # 原训练验证精度
    accuracies_test = torch.zeros(num_tasks, dtype=torch.float) # 原训练测试精度

    weight_mag_sum = torch.zeros((total_iters, num_hidden_layers + 1), dtype=torch.float) # 线性模型权重

    # 非线性模型参数
    rank_measure_period = 60000
    effective_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
    approximate_ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
    approximate_ranks_abs = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers),
                                        dtype=torch.float)
    ranks = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)
    dead_neurons = torch.zeros((int(total_examples / rank_measure_period), num_hidden_layers), dtype=torch.float)

    iter = 0
    is_checked = False

    # 数据集
    with open('data/mnist_', 'rb+') as f:
        x_train, y_train, x_test, y_test = pickle.load(f)
        if use_gpu == 1:
            x_train = x_train.to(dev)
            y_train = y_train.to(dev)
            x_test = x_test.to(dev)
            y_test = y_test.to(dev)

    permutation = load_data(file='data/permutation')

    task_permutations = permutation['task_permutations']  # 保存每个任务的像素排列
    data_permutations = permutation['data_permutations']

    # 原训练
    for task_idx in range(num_tasks):
        print(f"Task {task_idx}:")

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

        new_iter_start = iter

        x, y = x_train[:, task_permutations[task_idx]], y_train
        x, y = x[data_permutations[task_idx]], y[data_permutations[task_idx]]

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

            # learner.net.train()
            # train the network
            loss, network_output = learner.learn(x=batch_x, target=batch_y)

            if to_log and agent_type != 'linear':
                for idx, layer_idx in enumerate(learner.net.layers_to_log):
                    weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
            # log accuracy
            with torch.no_grad():
                accuracies[iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()
            iter += 1

        print('recent accuracy', accuracies[new_iter_start:iter - 1].mean())

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
            # save_data(file=params['data_file'], data=data)

        # 检验
        print('train accuracy: ')
        for t_idx in range(task_idx+1):
            with torch.no_grad():
                x_train_permuted = x_train[:, task_permutations[t_idx]]
                network_output, _ = learner.net.predict(x_train_permuted)
                train_accuracy = accuracy(softmax(network_output, dim=1), y_train).cpu()
            accuracies_train[t_idx] = train_accuracy
            print(f" Task {t_idx}: {accuracies_train[t_idx]:.4f}")
        
        print('train accuracy (mean): ', accuracies_train[:task_idx+1].mean())

        # 测试
        print('test accuracy: ')
        for t_idx in range(task_idx+1): # 测试不同序列
            with torch.no_grad():
                x_test_permuted = x_test[:, task_permutations[t_idx]]
                network_output, _ = learner.net.predict(x_test_permuted)
                test_accuracy = accuracy(softmax(network_output, dim=1), y_test).cpu()
            accuracies_test[t_idx] = test_accuracy
            print(f" Task {t_idx}: {accuracies_test[t_idx]:.4f}")
        
        print('test accuracy (mean): ', accuracies_test[:task_idx+1].mean())

    data = {
        'accuracies': accuracies.cpu(),
        'weight_mag_sum': weight_mag_sum.cpu(),
        'ranks': ranks.cpu(),
        'effective_ranks': effective_ranks.cpu(),
        'approximate_ranks': approximate_ranks.cpu(),
        'abs_approximate_ranks': approximate_ranks_abs.cpu(),
        'dead_neurons': dead_neurons.cpu(),
    }
    # save_data(file=params['data_file'], data=data)

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

    online_expr(params)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

import os
import random
import sys
import json
import torch
import pickle
import argparse
import numpy as np

seed = 42
torch.manual_seed(seed)                # PyTorch CPU 种子
torch.cuda.manual_seed(seed)           # PyTorch GPU 种子
torch.cuda.manual_seed_all(seed)       # 多GPU场景
np.random.seed(seed)                   # NumPy 种子
random.seed(seed)                      # Python 原生随机种子

def load_permutation(params: {}):  # 混合任务序列

    num_tasks = 200
    change_after = 10 * 6000
    if 'num_tasks' in params.keys():  # 顺序训练和联合训练共用同一参数（800）
        num_tasks = params['num_tasks']
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"] / params["change_after"])

    if 'change_after' in params.keys():
        change_after = params['change_after']

    num_tasks = 100 # 测试时缩小后的任务数

    classes_per_task = 10
    images_per_class = 6000
    input_size = 784  # 28x28

    examples_per_task = images_per_class * classes_per_task

    # permutation_1 = np.arange(examples_per_task)  # 任务次序（不需要）
    permutation = np.arange(input_size) # 打乱序列记录中间值
    task_permutations = []  # 保存每个任务的像素排列
    x_permutation_indices = []  # 保存每个任务的数据-对应排列
    data_permutations = []

    # 生成随机序列
    for task_idx in range(num_tasks):

        pixel_permutation = np.random.permutation(input_size)  # 每个任务不同的像素排列
        # data_permutation = np.random.permutation(examples_per_task)  # 打乱任务次序

        # permutation_1 = permutation_1[data_permutation]
        # data_permutations.append(permutation_1.copy())

        permutation = permutation[pixel_permutation]
        task_permutations.append(permutation.copy())
        for i in range(change_after):  # 有一个缺点就是选的数据初始x是一致的
            x_permutation_indices.append((i, task_idx))
        
    data = {
        # 'data_permutations': data_permutations,
        'task_permutations': task_permutations,
        'x_permutation_indices': x_permutation_indices
    } # 可以的话应该把联合训练的混合序列包括进去
    save_data(file='data/permutation', data=data) # 保存排列数据

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

    load_permutation(params)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

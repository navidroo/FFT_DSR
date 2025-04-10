import os
import csv
import random

import numpy as np
import torch

from random import randrange
import time


def to_cuda(sample):
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Running on CPU.")
        return sample
        
    print(f"Moving tensors to CUDA device: {torch.cuda.get_device_name(0)}")
    sampleout = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            sampleout[key] = val.cuda()
            print(f"  Moved tensor '{key}' to {sampleout[key].device}")
        elif isinstance(val, list):
            new_val = []
            for i, e in enumerate(val):
                if isinstance(e, torch.Tensor):
                    new_val.append(e.cuda())
                    print(f"  Moved list item {i} in '{key}' to {new_val[-1].device}")
                else:
                    new_val.append(val)
            sampleout[key] = new_val
        else:
            sampleout[key] = val
    return sampleout


def seed_all(seed):
    # Fix all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def new_log(folder_path, args=None):
    os.makedirs(folder_path, exist_ok=True)
    n_exp = len(os.listdir(folder_path))
    randn  = round((time.time()*1000000) % 1000)
    experiment_folder = os.path.join(folder_path, f'experiment_{n_exp}_{randn}')
    os.mkdir(experiment_folder)

    if args is not None:
        args_dict = args.__dict__
        write_params(args_dict, os.path.join(experiment_folder, 'args' + '.csv'))

    return experiment_folder, n_exp, randn


def write_params(params, path):
    with open(path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['key', 'value'])
        for data in params.items():
            writer.writerow([el for el in data])

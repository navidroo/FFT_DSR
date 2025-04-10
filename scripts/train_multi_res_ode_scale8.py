#!/usr/bin/env python3
"""
Training script for Multi-Resolution ODE-based Guided Anisotropic Diffusion
Optimized for scale=8 super-resolution
"""

import os
import sys
import subprocess

# Get the absolute path of the repository root directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the repository root to the Python path
sys.path.insert(0, REPO_ROOT)

def train_multi_res_ode(dataset='Middlebury', data_dir='./datafolder', save_dir='./save_dir',
                         num_epochs=4500, scaling=8, batch_size=4, crop_size=256, 
                         lr=0.0001, lr_step=100, val_every=10, in_memory=True, 
                         num_bands=3, ode_method='dopri5', block_size=128, overlap=32,
                         adaptive_block_size=True, use_wandb=False):
    """
    Train the Multi-Resolution ODE-based model on the specified dataset.
    
    Parameters:
    - dataset: Name of the dataset ('Middlebury', 'NYUv2', or 'DIML')
    - data_dir: Directory containing the datasets
    - save_dir: Directory to save models and logs
    - num_epochs: Number of training epochs
    - scaling: Super-resolution scaling factor (8 recommended for this model)
    - batch_size: Batch size for training
    - crop_size: Size of the input patches
    - lr: Learning rate
    - lr_step: Learning rate step size (epochs)
    - val_every: Validation interval (epochs)
    - in_memory: Whether to store dataset in memory
    - num_bands: Number of frequency bands
    - ode_method: ODE solver method
    - block_size: Block size for FFT processing
    - overlap: Overlap between blocks
    - adaptive_block_size: Whether to adapt block size based on scaling
    - use_wandb: Whether to use Weights & Biases for logging
    """
    
    # Set number of epochs and learning rate step based on dataset
    if dataset == 'Middlebury':
        num_epochs = num_epochs if num_epochs else 4500
        lr_step = lr_step if lr_step else 100
        val_every = val_every if val_every else 10
    elif dataset == 'NYUv2':
        num_epochs = num_epochs if num_epochs else 550
        lr_step = lr_step if lr_step else 10
        val_every = val_every if val_every else 4
    elif dataset == 'DIML':
        num_epochs = num_epochs if num_epochs else 300
        lr_step = lr_step if lr_step else 6
        val_every = val_every if val_every else 2
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Construct the command
    cmd = [
        'python', os.path.join(REPO_ROOT, 'run_train.py'),
        f'--dataset={dataset}',
        f'--data-dir={data_dir}',
        f'--save-dir={save_dir}',
        f'--num-epochs={num_epochs}',
        f'--scaling={scaling}',
        f'--batch-size={batch_size}',
        f'--crop-size={crop_size}',
        f'--lr={lr}',
        f'--lr-step={lr_step}',
        f'--val-every-n-epochs={val_every}',
        f'--block-size={block_size}',
        f'--overlap={overlap}',
        f'--num-bands={num_bands}',
        f'--ode-method={ode_method}',
        '--use-multi-res-ode'
    ]
    
    # Add optional flags
    if in_memory:
        cmd.append('--in-memory')
    if adaptive_block_size:
        cmd.append('--adaptive-block-size')
    if use_wandb:
        cmd.append('--wandb')
    
    # Print the command
    cmd_str = ' '.join(cmd)
    print(f"Running command: {cmd_str}")
    
    # Execute the command
    subprocess.run(cmd)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Multi-Resolution ODE-based DADA model')
    parser.add_argument('--dataset', type=str, default='Middlebury', choices=['Middlebury', 'NYUv2', 'DIML'],
                        help='Dataset to train on')
    parser.add_argument('--data-dir', type=str, default='./datafolder', 
                        help='Directory containing the datasets')
    parser.add_argument('--save-dir', type=str, default='./save_dir', 
                        help='Directory to save models and logs')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='Number of training epochs (defaults based on dataset)')
    parser.add_argument('--scaling', type=int, default=8,
                        help='Super-resolution scaling factor')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='Size of the input patches')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--lr-step', type=int, default=None,
                        help='Learning rate step size (defaults based on dataset)')
    parser.add_argument('--val-every', type=int, default=None,
                        help='Validation interval (defaults based on dataset)')
    parser.add_argument('--in-memory', action='store_true',
                        help='Store dataset in memory')
    parser.add_argument('--num-bands', type=int, default=3,
                        help='Number of frequency bands')
    parser.add_argument('--ode-method', type=str, default='dopri5', choices=['dopri5', 'rk4', 'euler'],
                        help='ODE solver method')
    parser.add_argument('--block-size', type=int, default=128,
                        help='Block size for FFT processing')
    parser.add_argument('--overlap', type=int, default=32,
                        help='Overlap between blocks')
    parser.add_argument('--adaptive-block-size', action='store_true',
                        help='Adapt block size based on scaling')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    train_multi_res_ode(
        dataset=args.dataset,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        num_epochs=args.num_epochs,
        scaling=args.scaling,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        lr=args.lr,
        lr_step=args.lr_step,
        val_every=args.val_every,
        in_memory=args.in_memory,
        num_bands=args.num_bands,
        ode_method=args.ode_method,
        block_size=args.block_size,
        overlap=args.overlap,
        adaptive_block_size=args.adaptive_block_size,
        use_wandb=args.wandb
    ) 
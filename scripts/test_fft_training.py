#!/usr/bin/env python
# Script to test FFT-based training with minimal iterations

import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
from collections import defaultdict
import traceback

from model import FFTGADBase
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from utils import to_cuda

def parse_args():
    parser = argparse.ArgumentParser(description='Test FFT-based training with minimal iterations')
    parser.add_argument('--dataset', type=str, default='middlebury', choices=['middlebury', 'nyuv2', 'diml'],
                        help='Dataset to use for testing')
    parser.add_argument('--data-dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--scale', type=int, default=8, choices=[4, 8, 16, 32],
                        help='Upsampling scale factor')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--Npre', type=int, default=10,
                        help='Number of iterations without gradient')
    parser.add_argument('--Ntrain', type=int, default=5,
                        help='Number of iterations with gradient')
    parser.add_argument('--block-size', type=int, default=64,
                        help='Block size for FFT processing')
    parser.add_argument('--overlap', type=int, default=16,
                        help='Overlap between blocks for FFT processing')
    return parser.parse_args()

def get_dataset(dataset_name, data_dir, scale):
    print(f"Creating {dataset_name} dataset with data_dir={data_dir}, scale={scale}")
    try:
        if dataset_name == 'middlebury':
            return MiddleburyDataset(data_dir=data_dir, split='test', scale=scale)
        elif dataset_name == 'nyuv2':
            return NYUv2Dataset(data_dir=data_dir, split='test', scale=scale)
        elif dataset_name == 'diml':
            return DIMLDataset(data_dir=data_dir, split='test', scale=scale)
        else:
            raise ValueError(f'Unknown dataset: {dataset_name}')
    except Exception as e:
        print(f"Error creating dataset: {e}")
        traceback.print_exc()
        raise

def test_fft_training():
    args = parse_args()
    
    print(f"Testing FFT-based training with dataset={args.dataset}, scale={args.scale}, "
          f"Npre={args.Npre}, Ntrain={args.Ntrain}, block_size={args.block_size}, overlap={args.overlap}")
    
    # Create dataset
    try:
        print("Creating dataset...")
        dataset = get_dataset(args.dataset, args.data_dir, args.scale)
        print(f"Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        traceback.print_exc()
        return
    
    # Create dataloader
    try:
        print("Creating dataloader...")
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        print(f"Dataloader created with {len(dataloader)} batches")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        traceback.print_exc()
        return
    
    # Create model
    try:
        print("Creating model...")
        model = FFTGADBase(
            feature_extractor='UNet',
            Npre=args.Npre,
            Ntrain=args.Ntrain,
            block_size=args.block_size,
            overlap=args.overlap
        ).cuda()
        print("Model created successfully")
    except Exception as e:
        print(f"Error creating model: {e}")
        traceback.print_exc()
        return
    
    # Create optimizer
    try:
        print("Creating optimizer...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        print("Optimizer created successfully")
    except Exception as e:
        print(f"Error creating optimizer: {e}")
        traceback.print_exc()
        return
    
    # Test forward and backward pass
    try:
        print("\nTesting forward and backward pass...")
        model.train()
        
        # Get a batch
        print("Getting a batch from dataloader...")
        for batch_idx, sample in enumerate(dataloader):
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}")
            if batch_idx > 0:
                break
                
            print("Moving batch to GPU...")
            sample = to_cuda(sample)
            
            print("Sample keys:", sample.keys())
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            
            # Forward pass
            print("Starting forward pass...")
            start_time = time.time()
            output = model(sample, train=True)
            forward_time = time.time() - start_time
            print(f"Forward pass completed in {forward_time:.2f} seconds")
            
            # Calculate loss
            print("Calculating loss...")
            pred = output['y_pred']
            target = sample['y_gt']
            loss = F.l1_loss(pred, target)
            print(f"Loss: {loss.item()}")
            
            # Backward pass
            print("Starting backward pass...")
            start_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - start_time
            print(f"Backward pass completed in {backward_time:.2f} seconds")
            
            print(f"Total time: {forward_time + backward_time:.2f} seconds")
            print("Test completed successfully!")
            return
            
    except Exception as e:
        print(f"Error during forward/backward pass: {e}")
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_fft_training() 
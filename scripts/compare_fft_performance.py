#!/usr/bin/env python
# Script to compare the performance of the original DADA model and the FFT-accelerated version

import os
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model import GADBase, FFTGADBase
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from utils import to_cuda

def parse_args():
    parser = argparse.ArgumentParser(description='Compare performance of original and FFT-accelerated DADA')
    parser.add_argument('--dataset', type=str, default='middlebury', choices=['middlebury', 'nyuv2', 'diml'],
                        help='Dataset to use for evaluation')
    parser.add_argument('--scale', type=int, default=8, choices=[4, 8, 16, 32],
                        help='Upsampling scale factor')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to evaluate')
    parser.add_argument('--original_iterations', type=int, default=8000,
                        help='Number of iterations for the original model')
    parser.add_argument('--fft_iterations', type=int, default=2000,
                        help='Number of iterations for the FFT-accelerated model')
    parser.add_argument('--block_size', type=int, default=64,
                        help='Block size for FFT processing')
    parser.add_argument('--overlap', type=int, default=16,
                        help='Overlap between blocks for FFT processing')
    return parser.parse_args()

def get_dataset(dataset_name, scale):
    if dataset_name == 'middlebury':
        return MiddleburyDataset(split='test', scale=scale)
    elif dataset_name == 'nyuv2':
        return NYUv2Dataset(split='test', scale=scale)
    elif dataset_name == 'diml':
        return DIMLDataset(split='test', scale=scale)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

def evaluate_model(model, dataloader, num_samples):
    model.eval()
    mse_values = []
    mae_values = []
    inference_times = []
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break
                
            sample = to_cuda(sample)
            
            # Measure inference time
            start_time = time.time()
            output = model(sample)
            torch.cuda.synchronize()
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Calculate metrics
            pred = output['y_pred']
            target = sample['y_gt']
            mask = sample['mask']
            
            # Apply mask if available
            if mask is not None:
                pred = pred * mask
                target = target * mask
            
            # Calculate MSE and MAE
            mse = F.mse_loss(pred, target).item() * 10000  # Convert to cm^2
            mae = F.l1_loss(pred, target).item() * 100     # Convert to cm
            
            mse_values.append(mse)
            mae_values.append(mae)
            
            print(f'Sample {i+1}: MSE = {mse:.2f} cm^2, MAE = {mae:.2f} cm, Inference Time = {inference_time:.2f} s')
    
    # Calculate average metrics
    avg_mse = np.mean(mse_values)
    avg_mae = np.mean(mae_values)
    avg_inference_time = np.mean(inference_times)
    
    return avg_mse, avg_mae, avg_inference_time, inference_times

def main():
    args = parse_args()
    
    # Create dataset and dataloader
    dataset = get_dataset(args.dataset, args.scale)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize models
    original_model = GADBase(feature_extractor='UNet', Npre=args.original_iterations, Ntrain=0).cuda()
    fft_model = FFTGADBase(feature_extractor='UNet', Npre=args.fft_iterations, Ntrain=0, 
                           block_size=args.block_size, overlap=args.overlap).cuda()
    
    # Evaluate original model
    print("\n===== Evaluating Original DADA Model =====")
    orig_mse, orig_mae, orig_time, orig_times = evaluate_model(original_model, dataloader, args.num_samples)
    
    # Evaluate FFT-accelerated model
    print("\n===== Evaluating FFT-Accelerated DADA Model =====")
    fft_mse, fft_mae, fft_time, fft_times = evaluate_model(fft_model, dataloader, args.num_samples)
    
    # Print comparison
    print("\n===== Performance Comparison =====")
    print(f"Original DADA ({args.original_iterations} iterations):")
    print(f"  Average MSE: {orig_mse:.2f} cm^2")
    print(f"  Average MAE: {orig_mae:.2f} cm")
    print(f"  Average Inference Time: {orig_time:.2f} s")
    
    print(f"\nFFT-Accelerated DADA ({args.fft_iterations} iterations):")
    print(f"  Average MSE: {fft_mse:.2f} cm^2")
    print(f"  Average MAE: {fft_mae:.2f} cm")
    print(f"  Average Inference Time: {fft_time:.2f} s")
    
    # Calculate speedup and quality change
    speedup = orig_time / fft_time
    mse_change = (fft_mse - orig_mse) / orig_mse * 100
    mae_change = (fft_mae - orig_mae) / orig_mae * 100
    
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"MSE Change: {mse_change:.2f}%")
    print(f"MAE Change: {mae_change:.2f}%")
    
    # Print individual sample times for detailed analysis
    print("\n===== Detailed Timing Analysis =====")
    print("Original model inference times (seconds):")
    for i, t in enumerate(orig_times):
        print(f"  Sample {i+1}: {t:.2f} s")
    
    print("\nFFT model inference times (seconds):")
    for i, t in enumerate(fft_times):
        print(f"  Sample {i+1}: {t:.2f} s")

if __name__ == '__main__':
    main() 
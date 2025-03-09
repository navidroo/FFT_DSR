#!/usr/bin/env python
# Script to test the FFT-based model

import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
import time

from model import FFTGADBase

def test_fft_model():
    print("Creating test data...")
    # Create a fake input
    batch_size = 1
    channels = 3
    height = 256
    width = 256
    source_h = 32
    source_w = 32
    
    # Create random guide and source
    guide = torch.rand(batch_size, channels, height, width).cuda()
    source = torch.rand(batch_size, 1, source_h, source_w).cuda()
    
    # Create bicubic upsampled initial guess
    initial = F.interpolate(source, size=(height, width), mode='bicubic').cuda()
    
    # Create mask
    mask_lr = torch.ones(batch_size, 1, source_h, source_w).cuda()
    
    # Create sample dictionary
    sample = {
        'guide': guide,
        'source': source,
        'y_bicubic': initial,
        'mask_lr': mask_lr,
    }
    
    # Test with different configurations
    test_configs = [
        {"Npre": 10, "Ntrain": 5, "block_size": 64, "overlap": 16},
        {"Npre": 50, "Ntrain": 10, "block_size": 32, "overlap": 8},
    ]
    
    for config in test_configs:
        print(f"\n\nTesting with config: {config}")
        
        # Instantiate the model
        model = FFTGADBase(
            Npre=config["Npre"],
            Ntrain=config["Ntrain"],
            block_size=config["block_size"],
            overlap=config["overlap"]
        ).cuda()
        
        # Forward pass
        try:
            print("Starting forward pass...")
            start_time = time.time()
            output = model(sample)
            end_time = time.time()
            
            print(f"Forward pass completed in {end_time - start_time:.2f} seconds")
            print(f"Output keys: {output.keys()}")
            print(f"Predicted shape: {output['y_pred'].shape}")
            
            # Check if output matches expected shape
            assert output['y_pred'].shape == (batch_size, 1, height, width), "Output shape mismatch"
            
            # Check if output has valid values
            assert not torch.isnan(output['y_pred']).any(), "Output contains NaN values"
            assert not torch.isinf(output['y_pred']).any(), "Output contains Inf values"
            
            print("Test passed!")
        except Exception as e:
            print(f"Test failed with error: {e}")

if __name__ == "__main__":
    test_fft_model() 
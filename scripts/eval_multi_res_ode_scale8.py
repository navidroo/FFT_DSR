#!/usr/bin/env python3
"""
Evaluation script for Multi-Resolution ODE-based Guided Anisotropic Diffusion
Optimized for scale=8 super-resolution
"""

import os
import sys
import subprocess
import json

# Get the absolute path of the repository root directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the repository root to the Python path
sys.path.insert(0, REPO_ROOT)

def evaluate_multi_res_ode(checkpoint, dataset='Middlebury', data_dir='./datafolder', 
                           scaling=8, batch_size=1, crop_size=1024, 
                           num_bands=3, ode_method='dopri5', block_size=128, overlap=32,
                           adaptive_block_size=True, in_memory=False):
    """
    Evaluate the Multi-Resolution ODE-based model on the specified dataset.
    
    Parameters:
    - checkpoint: Path to the model checkpoint
    - dataset: Name of the dataset ('Middlebury', 'NYUv2', or 'DIML')
    - data_dir: Directory containing the datasets
    - scaling: Super-resolution scaling factor (8 recommended for this model)
    - batch_size: Batch size for evaluation
    - crop_size: Size of the input patches
    - num_bands: Number of frequency bands
    - ode_method: ODE solver method
    - block_size: Block size for FFT processing
    - overlap: Overlap between blocks
    - adaptive_block_size: Whether to adapt block size based on scaling
    - in_memory: Whether to store dataset in memory
    
    Returns:
    - Dictionary with evaluation metrics
    """
    
    # Set optimal Npre and Ntrain parameters for evaluation (can be lower than training)
    if dataset == 'Middlebury':
        Npre, Ntrain = 1000, 128
    elif dataset == 'NYUv2':
        Npre, Ntrain = 800, 64
    elif dataset == 'DIML':
        Npre, Ntrain = 600, 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create a temporary file to store the evaluation results
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        results_file = temp_file.name
    
    # Construct the command
    cmd = [
        'python', os.path.join(REPO_ROOT, 'run_eval.py'),
        f'--checkpoint={checkpoint}',
        f'--dataset={dataset}',
        f'--data-dir={data_dir}',
        f'--scaling={scaling}',
        f'--batch-size={batch_size}',
        f'--crop-size={crop_size}',
        f'--Npre={Npre}',
        f'--Ntrain={Ntrain}',
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
    
    # Print the command
    cmd_str = ' '.join(cmd)
    print(f"Running command: {cmd_str}")
    
    # Execute the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse and return the evaluation metrics
    try:
        # Try to extract JSON metrics from the output
        output = result.stdout
        
        # Find all lines that look like JSON dictionaries
        import re
        json_lines = re.findall(r'({.*})', output)
        
        if json_lines:
            # Parse the last JSON dictionary found
            metrics = json.loads(json_lines[-1])
            
            # Print the metrics in a readable format
            print("\nEvaluation Results:")
            print("=" * 40)
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.6f}")
                else:
                    print(f"{key}: {value}")
            print("=" * 40)
            
            return metrics
        else:
            print("No metrics found in the output. Raw output:")
            print(output)
            return {}
    except Exception as e:
        print(f"Error parsing evaluation results: {e}")
        print("Raw output:")
        print(result.stdout)
        print("Error output:")
        print(result.stderr)
        return {}

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Multi-Resolution ODE-based DADA model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--dataset', type=str, default='Middlebury', choices=['Middlebury', 'NYUv2', 'DIML'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data-dir', type=str, default='./datafolder', 
                        help='Directory containing the datasets')
    parser.add_argument('--scaling', type=int, default=8,
                        help='Super-resolution scaling factor')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='Size of the input patches')
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
    
    args = parser.parse_args()
    
    evaluate_multi_res_ode(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        data_dir=args.data_dir,
        scaling=args.scaling,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        num_bands=args.num_bands,
        ode_method=args.ode_method,
        block_size=args.block_size,
        overlap=args.overlap,
        adaptive_block_size=args.adaptive_block_size,
        in_memory=args.in_memory
    ) 
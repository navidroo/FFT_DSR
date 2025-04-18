import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path to evaluate')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--data-dir', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--crop-size', type=int, default=256, help='Size of the input (squared) patches')
parser.add_argument('--scaling', type=int, default=8, help='Scaling factor')
parser.add_argument('--in-memory', default=False, action='store_true', help='Hold data in memory during evaluation')
parser.add_argument('--no_params', default=False, action='store_true', help='Hold data in memory during evaluation')
parser.add_argument('--feature-extractor', type=str, default='UNet', help='Feature extractor for edge potentials')

parser.add_argument('--Npre', type=int, default=8000, help='N learned iterations, but without gradients')
parser.add_argument('--Ntrain', type=int, default=1024, help='N learned iterations with gradients')

# FFT-accelerated model arguments
parser.add_argument('--use-fft', action='store_true', default=False, help='Use FFT-accelerated model')
parser.add_argument('--block-size', type=int, default=64, help='Block size for FFT acceleration')
parser.add_argument('--overlap', type=int, default=16, help='Overlap size for FFT blocks')
parser.add_argument('--adaptive-block-size', action='store_true', default=False, 
                    help='Adaptively set block size based on scaling factor (base: 64 for 4x scaling)')

# Multi-Resolution ODE parameters
parser.add_argument('--use-multi-res-ode', action='store_true', default=False, 
                    help='Use Multi-Resolution ODE-based diffusion (recommended for scale=8)')
parser.add_argument('--num-bands', type=int, default=3, 
                    help='Number of frequency bands for multi-resolution approach')
parser.add_argument('--ode-rtol', type=float, default=1e-3, 
                    help='Relative tolerance for ODE solver')
parser.add_argument('--ode-atol', type=float, default=1e-3, 
                    help='Absolute tolerance for ODE solver')
parser.add_argument('--ode-method', type=str, default='dopri5', 
                    choices=['dopri5', 'rk4', 'euler'], 
                    help='ODE solver method')

# SwinFuSR-specific parameters
parser.add_argument('--use-swinfusr', action='store_true', default=False, 
                    help='Use SwinFuSR-GAD model with Swin Transformers and cross-domain fusion')
parser.add_argument('--swin-window-size', type=int, default=7, 
                    help='Window size for Swin Transformer attention')
parser.add_argument('--num-heads', type=int, default=3, 
                    help='Number of attention heads in Swin Transformer')
parser.add_argument('--num-swin-blocks', type=int, default=4, 
                    help='Number of Swin Transformer blocks')
import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

# general
parser.add_argument('--save-dir', required=True, help='Path to directory where models and logs should be saved')
parser.add_argument('--logstep-train', default=10, type=int, help='Training log interval in steps')
parser.add_argument('--save-model', default='both', choices=['last', 'best', 'no', 'both'])
parser.add_argument('--val-every-n-epochs', type=int, default=1, help='Validation interval in epochs')
parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume')
parser.add_argument('--seed', type=int, default=12345, help='Random seed')
parser.add_argument('--wandb', action='store_true', default=False, help='Use Weights & Biases instead of TensorBoard')
parser.add_argument('--wandb-project', type=str, default='DADA-SR', help='Wandb project name')

# data
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--data-dir', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--crop-size', type=int, default=256, help='Size of the input (squared) patches')
parser.add_argument('--scaling', type=int, default=8, help='Scaling factor')
parser.add_argument('--max-rotation', type=float, default=15., help='Maximum rotation angle (degrees)')
parser.add_argument('--no-flip', action='store_true', default=False, help='Switch off random flipping')
parser.add_argument('--in-memory', action='store_true', default=False, help='Hold data in memory during training')

# training
parser.add_argument('--loss', default='l1', type=str, choices=['l1'])
parser.add_argument('--num-epochs', type=int, default=4500) 
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--w-decay', type=float, default=1e-5)
parser.add_argument('--lr-scheduler', type=str, default='step', choices=['no', 'step', 'plateau'])
parser.add_argument('--lr-step', type=int, default=100, help='LR scheduler step size (epochs)')
parser.add_argument('--lr-gamma', type=float, default=0.9, help='LR decay rate')
parser.add_argument('--skip-first', action='store_true', help='Don\'t optimize during first epoch')
parser.add_argument('--gradient-clip', type=float, default=0.01, help='If > 0, clips gradient norm to that value')
parser.add_argument('--no-opt', action='store_true', help='Don\'t optimize')

# model
parser.add_argument('--feature-extractor', type=str, default='UNet', help="Feature extractor for edge potentials. 'none' for the unlearned version.") 
parser.add_argument('--Npre', type=int, default=8000, help='N learned iterations, but without gradients')
parser.add_argument('--Ntrain', type=int, default=1024, help='N learned iterations with gradients')

# FFT-specific parameters
parser.add_argument('--use-fft', action='store_true', default=False, help='Use FFT-accelerated diffusion')
parser.add_argument('--block-size', type=int, default=64, help='Block size for FFT processing')
parser.add_argument('--overlap', type=int, default=16, help='Overlap between blocks for FFT processing')
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

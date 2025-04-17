import os
import argparse
from collections import defaultdict
import time

import torch
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import eval_parser
from model import GADBase, FFTGADBase, MultiResODEGAD, SwinFuSRGAD
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from utils import to_cuda

from losses import get_loss
import time

# Function to get GPU memory usage
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        # Returns GPU memory usage in MB
        return torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        return 0, 0


class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloader = self.get_dataloader(args)
        
        # Reset CUDA memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        # Choose model type based on arguments
        if args.use_swinfusr:
            print("Using SwinFuSR-GAD model for evaluation...")
            self.model = SwinFuSRGAD(
                feature_extractor='SwinFuSR',
                Npre=args.Npre,
                Ntrain=args.Ntrain,
                block_size=args.block_size,
                overlap=args.overlap,
                adaptive_block_size=args.adaptive_block_size,
                scaling_factor=args.scaling,
                swin_window_size=args.swin_window_size,
                num_heads=args.num_heads,
                num_swin_blocks=args.num_swin_blocks
            )
        elif args.use_multi_res_ode:
            print("Using Multi-Resolution ODE-based model for evaluation...")
            self.model = MultiResODEGAD( 
                args.feature_extractor, 
                Npre=args.Npre,
                Ntrain=args.Ntrain,
                block_size=args.block_size,
                overlap=args.overlap,
                adaptive_block_size=args.adaptive_block_size,
                scaling_factor=args.scaling,
                bands=args.num_bands,
                ode_rtol=args.ode_rtol,
                ode_atol=args.ode_atol,
                ode_method=args.ode_method
            )
        elif args.use_fft:
            print("Using FFT-accelerated model for evaluation...")
            self.model = FFTGADBase(
                args.feature_extractor, 
                Npre=args.Npre, 
                Ntrain=args.Ntrain,
                block_size=args.block_size,
                overlap=args.overlap,
                adaptive_block_size=args.adaptive_block_size,
                scaling_factor=args.scaling
            )
        else:
            print("Using standard model for evaluation...")
            self.model = GADBase(args.feature_extractor, Npre=args.Npre, Ntrain=args.Ntrain)
            
        self.resume(path=args.checkpoint)
        self.model.cuda().eval()

    def evaluate(self):
        test_stats = defaultdict(float)
        total_images = 0
        batch_times = []
        batch_sizes = []
        # Track memory stats
        memory_usage = []

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloader, leave=False):
                batch_size = sample['guide'].shape[0]
                batch_sizes.append(batch_size)
                total_images += batch_size
                
                sample = to_cuda(sample)
                
                # Record memory before processing
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                # Start timing for this batch
                batch_start = time.time()
                
                output = self.model(sample)
                _, loss_dict = get_loss(output, sample)
                
                # End timing for this batch
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Record peak memory usage for this batch
                if torch.cuda.is_available():
                    current_mem, peak_mem = get_gpu_memory_usage()
                    memory_usage.append(peak_mem)

                for key in loss_dict:
                    test_stats[key] += loss_dict[key]

        # Calculate timing statistics
        total_batch_time = sum(batch_times)
        avg_time_per_image = total_batch_time / total_images
        
        # Calculate weighted statistics for batch processing times
        # This accounts for the last batch potentially having fewer images
        weighted_times = [batch_times[i] / batch_sizes[i] for i in range(len(batch_times))]
        min_time_per_image = min(weighted_times)
        max_time_per_image = max(weighted_times)
        
        # Calculate memory statistics
        if memory_usage:
            avg_memory = sum(memory_usage) / len(memory_usage)
            max_memory = max(memory_usage)
        else:
            avg_memory = max_memory = 0
        
        # Add timing information to the statistics
        test_stats['total_batch_time'] = total_batch_time
        test_stats['avg_time_per_image'] = avg_time_per_image
        test_stats['min_time_per_image'] = min_time_per_image
        test_stats['max_time_per_image'] = max_time_per_image
        test_stats['total_images'] = total_images
        test_stats['num_batches'] = len(batch_times)
        test_stats['avg_memory_mb'] = avg_memory
        test_stats['max_memory_mb'] = max_memory

        # Only normalize quality metrics, not timing metrics
        timing_keys = ['total_batch_time', 'avg_time_per_image', 'min_time_per_image', 
                       'max_time_per_image', 'total_images', 'num_batches',
                       'avg_memory_mb', 'max_memory_mb']
        return {k: v / len(self.dataloader) if k not in timing_keys else v 
                for k, v in test_stats.items()}

    @staticmethod
    def get_dataloader(args: argparse.Namespace):
        data_args = {
            'crop_size': (args.crop_size, args.crop_size),
            'in_memory': args.in_memory,
            'max_rotation_angle': 0,
            'do_horizontal_flip': False,
            'crop_valid': True,
            'crop_deterministic': True,
            'image_transform': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            'scaling': args.scaling
        }

        if args.dataset == 'DIML':
            # depth_transform = Normalize([2749.64], [1154.29])
            depth_transform = Normalize([0.0], [1154.29])
            dataset = DIMLDataset(os.path.join(args.data_dir, 'DIML'), **data_args, split='test',
                                  depth_transform=depth_transform)
        elif args.dataset == 'Middlebury':
            # depth_transform = Normalize([2296.78], [1122.7])
            depth_transform = Normalize([0.0], [1122.7])
            dataset = MiddleburyDataset(os.path.join(args.data_dir, 'Middlebury'), **data_args, split='test',
                                        depth_transform=depth_transform)
        elif args.dataset == 'NYUv2':
            # depth_transform = Normalize([2796.32], [1386.05])
            depth_transform = Normalize([0.0], [1386.05])
            dataset = NYUv2Dataset(os.path.join(args.data_dir, 'NYU Depth v2'), **data_args, split='test',
                                   depth_transform=depth_transform)
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path)
        
        if 'model' in checkpoint:
            model_dict = checkpoint['model']
            # Handle potential incompatibilities between model types
            model_dict.pop('logk2', None)  # in case of using the old codebase, pop unnecessary keys
            model_dict.pop('mean_guide', None)
            model_dict.pop('std_guide', None)
            
            # Check if we're trying to load an FFT model into a non-FFT model or vice versa
            is_fft_checkpoint = any(k.startswith('feature_extraction.1.fft_') for k in model_dict.keys())
            is_fft_model = isinstance(self.model, FFTGADBase)
            
            if is_fft_checkpoint != is_fft_model:
                print(f"Warning: The checkpoint is {'an FFT' if is_fft_checkpoint else 'a non-FFT'} model, " 
                      f"but the configured model is {'an FFT' if is_fft_model else 'a non-FFT'} model. "
                      f"This might cause loading issues.")
            
            try:
                self.model.load_state_dict(model_dict)
                print(f'Checkpoint \'{path}\' loaded successfully.')
            except Exception as e:
                print(f'Error loading checkpoint: {e}')
                print('Attempting to load with strict=False...')
                self.model.load_state_dict(model_dict, strict=False)
                print(f'Checkpoint \'{path}\' loaded with strict=False.')
        else:
            try:
                self.model.load_state_dict(checkpoint)
                print(f'Checkpoint \'{path}\' loaded successfully.')
            except Exception as e:
                print(f'Error loading checkpoint: {e}')
                print('Attempting to load with strict=False...')
                self.model.load_state_dict(checkpoint, strict=False)
                print(f'Checkpoint \'{path}\' loaded with strict=False.')

    def transfer(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path)
        
        if 'model' in checkpoint:
            model_dict = checkpoint['model']
            # Handle potential incompatibilities between model types
            model_dict.pop('logk2', None)  # in case of using the old codebase, pop unnecessary keys
            model_dict.pop('mean_guide', None)
            model_dict.pop('std_guide', None)
            
            # Check if we're trying to load an FFT model into a non-FFT model or vice versa
            is_fft_checkpoint = any(k.startswith('feature_extraction.1.fft_') for k in model_dict.keys())
            is_fft_model = isinstance(self.model, FFTGADBase)
            
            if is_fft_checkpoint != is_fft_model:
                print(f"Warning: The checkpoint is {'an FFT' if is_fft_checkpoint else 'a non-FFT'} model, " 
                      f"but the configured model is {'an FFT' if is_fft_model else 'a non-FFT'} model. "
                      f"This might cause loading issues.")
            
            try:
                self.model.load_state_dict(model_dict)
                print(f'Checkpoint \'{path}\' transferred successfully.')
            except Exception as e:
                print(f'Error transferring checkpoint: {e}')
                print('Attempting to transfer with strict=False...')
                self.model.load_state_dict(model_dict, strict=False)
                print(f'Checkpoint \'{path}\' transferred with strict=False.')
        else:
            try:
                self.model.load_state_dict(checkpoint)
                print(f'Checkpoint \'{path}\' transferred successfully.')
            except Exception as e:
                print(f'Error transferring checkpoint: {e}')
                print('Attempting to transfer with strict=False...')
                self.model.load_state_dict(checkpoint, strict=False)
                print(f'Checkpoint \'{path}\' transferred with strict=False.')


if __name__ == '__main__':
    args = eval_parser.parse_args()
    print(eval_parser.format_values())
    
    # Print information about which model will be used
    if args.use_fft:
        if args.adaptive_block_size:
            base_block_size = args.block_size
            adjusted_block_size = int(base_block_size * (args.scaling / 4))
            actual_block_size = max(base_block_size, min(adjusted_block_size, 256))
            print(f"Using FFT-accelerated model with ADAPTIVE block sizing:")
            print(f"  - Base block size: {base_block_size}")
            print(f"  - Scaling factor: {args.scaling}")
            print(f"  - Calculated block size: {actual_block_size}")
            print(f"  - Adjusted overlap: {int(args.overlap * (actual_block_size / base_block_size))}")
        else:
            print(f"Using FFT-accelerated model with fixed block_size={args.block_size}, overlap={args.overlap}")
    else:
        print("Using standard non-FFT model")
    
    print(f"Model settings: Npre={args.Npre}, Ntrain={args.Ntrain}")

    evaluator = Evaluator(args)

    print("\nStarting evaluation...")
    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    # de-standardize losses and convert to cm (cm^2, respectively)
    std = evaluator.dataloader.dataset.depth_transform.std[0]
    stats['l1_loss'] = 0.1 * std * stats['l1_loss']
    stats['mse_loss'] = 0.01 * std**2 * stats['mse_loss']

    print('\n====== Evaluation Results ======')
    print(f"Total time: {time_elapsed:.2f}s ({time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s)")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Number of batches: {stats['num_batches']}")
    print(f"Batch size: {args.batch_size}")
    
    print('\n====== Timing Statistics ======')
    print(f"Average time per image: {stats['avg_time_per_image']*1000:.2f}ms ({stats['avg_time_per_image']:.4f}s)")
    print(f"Minimum time per image: {stats['min_time_per_image']*1000:.2f}ms")
    print(f"Maximum time per image: {stats['max_time_per_image']*1000:.2f}ms")
    print(f"Images per second: {1.0/stats['avg_time_per_image']:.2f}")
    
    print('\n====== Memory Statistics ======')
    print(f"Average GPU memory usage: {stats['avg_memory_mb']:.2f} MB")
    print(f"Maximum GPU memory usage: {stats['max_memory_mb']:.2f} MB")
    
    print('\n====== Quality Metrics ======')
    for key, value in stats.items():
        if key not in ['avg_time_per_image', 'min_time_per_image', 'max_time_per_image', 
                       'total_images', 'total_batch_time', 'num_batches', 
                       'avg_memory_mb', 'max_memory_mb']:
            print(f"  {key}: {value:.6f}")
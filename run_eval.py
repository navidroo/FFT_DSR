import os
import argparse
from collections import defaultdict
import time

import torch
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import eval_parser
from model import GADBase, FFTGADBase
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from utils import to_cuda

from losses import get_loss
import time


class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloader = self.get_dataloader(args)
        
        # Choose between regular GADBase and FFT-accelerated version
        if args.use_fft:
            print("Using FFT-accelerated model for evaluation...")
            self.model = FFTGADBase(
                args.feature_extractor, 
                Npre=args.Npre, 
                Ntrain=args.Ntrain,
                block_size=args.block_size,
                overlap=args.overlap
            )
        else:
            print("Using standard model for evaluation...")
            self.model = GADBase(args.feature_extractor, Npre=args.Npre, Ntrain=args.Ntrain)
            
        self.resume(path=args.checkpoint)
        self.model.cuda().eval()

    def evaluate(self):
        test_stats = defaultdict(float)
        num_samples = 0 

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloader, leave=False):
                sample = to_cuda(sample)
                
                output = self.model(sample)
                _, loss_dict = get_loss(output, sample)

                for key in loss_dict:
                    test_stats[key] += loss_dict[key]

        return {k: v / len(self.dataloader) for k, v in test_stats.items()}

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
        print(f"Using FFT-accelerated model with block_size={args.block_size}, overlap={args.overlap}")
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

    print('\nEvaluation completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f"Average time per sample: {time_elapsed / len(evaluator.dataloader):.4f}s")
    print("\nResults:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
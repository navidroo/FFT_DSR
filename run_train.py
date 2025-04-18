import os
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
from torch import is_tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm

from arguments import train_parser
from model import GADBase, FFTGADBase, MultiResODEGAD, SwinFuSRGAD
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from losses import get_loss
from utils import new_log, to_cuda, seed_all

# import nvidia_smi
# nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.use_wandb = self.args.wandb

        self.dataloaders = self.get_dataloaders(args)
        
        seed_all(args.seed)

        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA is not available! Training will be extremely slow on CPU.")
            
        # Choose model type based on arguments
        if args.use_swinfusr:
            print("Creating SwinFuSR-GAD model...")
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
            print("Creating Multi-Resolution ODE-based model...")
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
            print("Creating FFT-accelerated model...")
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
            print("Creating standard model...")
            self.model = GADBase( 
                args.feature_extractor, 
                Npre=args.Npre,
                Ntrain=args.Ntrain, 
            )
            
        # Move model to GPU if available
        if torch.cuda.is_available():
            print("Moving model to GPU...")
            self.model = self.model.cuda()
            # Verify model is on GPU
            for name, param in self.model.named_parameters():
                print(f"Parameter '{name}' is on device: {param.device}")
        else:
            print("Model will run on CPU.")

        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, args.dataset), args)
        self.args.experiment_folder = self.experiment_folder

        if self.use_wandb:
            import wandb
            wandb.init(project=args.wandb_project, dir=self.experiment_folder)
            wandb.config.update(self.args)
            self.writer = None
        # else:
            # self.writer = SummaryWriter(log_dir=self.experiment_folder)

        if not args.no_opt:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        else:
            self.optimizer = None
            self.scheduler = None

        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        if args.resume is not None:
            self.resume(path=args.resume)

    def __del__(self):
        if not self.use_wandb:
            self.writer.close()

    def train(self):
        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_scheduler == 'step':
                    if not args.no_opt:
                        self.scheduler.step()
                        if self.use_wandb:
                            wandb.log({'log_lr': np.log10(self.scheduler.get_last_lr())}, self.iter)
                        else:
                            self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch)

                self.epoch += 1

    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)

        self.model.train()
        
        # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # self.train_stats["gpu_used"] = info.used


        with tqdm(self.dataloaders['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            for i, sample in enumerate(inner_tnr):
                sample = to_cuda(sample)

                if not args.no_opt:
                    self.optimizer.zero_grad()

                output = self.model(sample, train=True)

                loss, loss_dict = get_loss(output, sample)

                if torch.isnan(loss):
                    raise Exception("detected NaN loss..")
                    
                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

                if self.epoch > 0 or not self.args.skip_first:
                    if not args.no_opt:
                        loss.backward()

                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                    if not args.no_opt:
                        self.optimizer.step()

                self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                        validation_loss=self.val_stats['optimization_loss'],
                                        best_validation_loss=self.best_optimization_loss)

                    if self.use_wandb:
                        wandb.log({k + '/train': v for k, v in self.train_stats.items()}, self.iter)
                    else:
                        for key in self.train_stats:
                            self.writer.add_scalar('train/' + key, self.train_stats[key], self.iter)

                    # reset metrics
                    self.train_stats = defaultdict(float)

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloaders['val'], leave=False):
                sample = to_cuda(sample)

                output = self.model(sample)

                loss, loss_dict = get_loss(output, sample)

                for key in loss_dict:
                    self.val_stats[key] +=  loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            if self.use_wandb:
                wandb.log({k + '/val': v for k, v in self.val_stats.items()}, self.iter)
            else:
                for key in self.val_stats:
                    self.writer.add_scalar('val/' + key, self.val_stats[key], self.epoch)

            if self.val_stats['optimization_loss'] < self.best_optimization_loss:
                self.best_optimization_loss = self.val_stats['optimization_loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')

    @staticmethod
    def get_dataloaders(args):
        data_args = {
            'crop_size': (args.crop_size, args.crop_size),
            'in_memory': args.in_memory,
            'max_rotation_angle': args.max_rotation,
            'do_horizontal_flip': not args.no_flip,
            'crop_valid': True,
            'image_transform': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            'scaling': args.scaling
        }

        phases = ('train', 'val')
        if args.dataset == 'Middlebury':
            # Important, do not zero-center the depth, DADA needs positive depths
            depth_transform = Normalize([0.0], [1122.7]) 
            datasets = {phase: MiddleburyDataset(os.path.join(args.data_dir, 'Middlebury'), **data_args, split=phase,
                        depth_transform=depth_transform, crop_deterministic=phase == 'val') for phase in phases}

        elif args.dataset == 'DIML':
            # Important, do not zero-center the depth, DADA needs positive depths
            depth_transform = Normalize([0.0], [1154.29])
            datasets = {phase: DIMLDataset(os.path.join(args.data_dir, 'DIML'), **data_args, split=phase,
                        depth_transform=depth_transform) for phase in phases}

        elif args.dataset == 'NYUv2':
            # Important, do not zero-center the depth, DADA needs positive depths
            depth_transform = Normalize([0.0], [1386.05])
            datasets = {phase: NYUv2Dataset(os.path.join(args.data_dir, 'NYU Depth v2'), **data_args, split=phase,
                        depth_transform=depth_transform) for phase in phases}
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return {phase: DataLoader(datasets[phase], batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=True, drop_last=False) for phase in phases}

    def save_model(self, prefix=''):
        if args.no_opt:
            torch.save({
                'model': self.model.state_dict(),
                'epoch': self.epoch + 1,
                'iter': self.iter
            }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))
        else:
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch + 1,
                'iter': self.iter
            }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if not args.no_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = train_parser.parse_args()
    print(train_parser.format_values())

    # Print information about model configuration
    print(f"\nTraining with dataset: {args.dataset}, scaling factor: {args.scaling}")
    
    if args.use_fft:
        if args.adaptive_block_size:
            base_block_size = args.block_size
            adjusted_block_size = int(base_block_size * (args.scaling / 4))
            actual_block_size = max(base_block_size, min(adjusted_block_size, 256))
            print(f"\nUsing FFT-accelerated model with ADAPTIVE block sizing:")
            print(f"  - Base block size: {base_block_size}")
            print(f"  - Scaling factor: {args.scaling}")
            print(f"  - Calculated block size: {actual_block_size}")
            print(f"  - Adjusted overlap: {int(args.overlap * (actual_block_size / base_block_size))}")
        else:
            print(f"\nUsing FFT-accelerated model with fixed block_size={args.block_size}, overlap={args.overlap}")
    else:
        print("\nUsing standard non-FFT model")
    
    print(f"Model settings: Npre={args.Npre}, Ntrain={args.Ntrain}")

    if args.wandb:
        import wandb

    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

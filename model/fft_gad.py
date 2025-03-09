# FFT-based Guided Anisotropic Diffusion for Accelerated Depth Super-Resolution
# Based on the original implementation by Nando Metzger

from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np

from model.fft_diffusion import FFTDiffuseBase, c, g, diffuse_step

INPUT_DIM = 4
FEATURE_DIM = 64

class FFTGADBase(nn.Module):
    
    def __init__(
            self, feature_extractor='UNet',
            Npre=2000, Ntrain=256,  # Reduced iterations due to FFT acceleration
            block_size=64, overlap=16,
    ):
        super().__init__()

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
        self.block_size = block_size
        self.overlap = overlap
 
        if feature_extractor=='none': 
            # RGB verion of DADA does not need a deep feature extractor
            self.feature_extractor = None
            self.Ntrain = 0
            self.logk = torch.log(torch.tensor(0.03))

        elif feature_extractor=='UNet':
            # Learned verion of DADA
            self.feature_extractor = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bicubic'),
                smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=INPUT_DIM),
                torch.nn.AvgPool2d(kernel_size=2, stride=2)
            )
            self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))

        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')
             
    def forward(self, sample, train=False, deps=0.1):
        guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']

        # assert that all values are positive, otherwise shift depth map to positives
        if source.min() <= deps:
            print("Warning: The forward function was called with negative depth values. Values were temporarly shifted. Consider using unnormalized depth values for stability.")
            source += deps
            sample['y_bicubic'] += deps
            shifted = True
        else:
            shifted = False

        y_pred, aux = self.diffuse(sample['y_bicubic'].clone(), guide.clone(), source, mask_lr < 0.5,
                 K=torch.exp(self.logk), verbose=False, train=train)

        # revert the shift
        if shifted:
            y_pred -= deps

        return {**{'y_pred': y_pred}, **aux}

    def diffuse(self, img, guide, source, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False):

        _,_,h,w = guide.shape
        _,_,sh,sw = source.shape

        # Define Downsampling operations that depend on the input size
        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')

        # Deep Learning version or RGB version to calculate the coefficients
        if self.feature_extractor is None: 
            guide_feats = torch.cat([guide, img], 1) 
        else:
            guide_feats = self.feature_extractor(torch.cat([guide, img-img.mean((1,2,3), keepdim=True)], 1))
        
        # Convert the features to coefficients with the Perona-Malik edge-detection function
        cv, ch = c(guide_feats, K=K)
        
        # Identify regions with relatively uniform diffusion coefficients
        # These are regions where FFT-based diffusion can be applied
        uniform_regions = self.identify_uniform_regions(cv, ch)

        # Iterations without gradient
        if self.Npre > 0: 
            with torch.no_grad():
                Npre = randrange(self.Npre) if train else self.Npre
                
                # Apply FFT diffusion to uniform regions first
                img = self.fft_diffuse(img, cv, ch, uniform_regions, l=l)
                
                # Then apply standard diffusion for remaining iterations
                for t in range(min(500, Npre)):                     
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, eps=eps)

        # Iterations with gradient
        if self.Ntrain > 0: 
            for t in range(self.Ntrain): 
                img = diffuse_step(cv, ch, img, l=l)
                img = adjust_step(img, source, mask_inv, upsample, downsample, eps=eps)

        return img, {"cv": cv, "ch": ch}
    
    def identify_uniform_regions(self, cv, ch, threshold=0.1):
        """
        Identify regions with relatively uniform diffusion coefficients.
        Returns a mask where 1 indicates uniform regions suitable for FFT.
        """
        # Calculate local variance of diffusion coefficients
        cv_var = F.avg_pool2d(cv**2, 3, stride=1, padding=1) - F.avg_pool2d(cv, 3, stride=1, padding=1)**2
        ch_var = F.avg_pool2d(ch**2, 3, stride=1, padding=1) - F.avg_pool2d(ch, 3, stride=1, padding=1)**2
        
        # Regions with low variance are considered uniform
        uniform_regions = (cv_var < threshold) & (ch_var < threshold)
        
        return uniform_regions
    
    def fft_diffuse(self, depth, cv, ch, uniform_regions, l=0.24, fft_steps=10):
        """
        Apply FFT-based diffusion to accelerate the process in uniform regions.
        """
        batch_size, channels, height, width = depth.shape
        
        # Process the image in blocks
        for b in range(batch_size):
            for y in range(0, height, self.block_size - self.overlap):
                for x in range(0, width, self.block_size - self.overlap):
                    # Define block boundaries with overlap
                    y_end = min(y + self.block_size, height)
                    x_end = min(x + self.block_size, width)
                    y_start = max(0, y)
                    x_start = max(0, x)
                    
                    # Extract block
                    block = depth[b:b+1, :, y_start:y_end, x_start:x_end]
                    
                    # Handle edge cases for diffusion coefficients
                    cv_end_y = min(y_end-1, cv.shape[2])
                    ch_end_x = min(x_end-1, ch.shape[3])
                    
                    block_cv = cv[b:b+1, :, y_start:cv_end_y, x_start:x_end]
                    block_ch = ch[b:b+1, :, y_start:y_end, x_start:ch_end_x]
                    block_uniform = uniform_regions[b:b+1, :, y_start:y_end, x_start:x_end]
                    
                    # If the block is mostly uniform, apply FFT diffusion
                    if block_uniform.float().mean() > 0.7:
                        # Apply FFT-based diffusion
                        block = self.fft_diffuse_block(block, block_cv, block_ch, l, fft_steps)
                        
                        # Apply blending for overlap regions to avoid boundary artifacts
                        if y > 0 or x > 0:
                            # Create blending weights for smooth transition
                            blend_y = torch.ones_like(block)
                            blend_x = torch.ones_like(block)
                            
                            if y > 0:
                                for i in range(min(self.overlap, y_end - y_start)):
                                    blend_y[:, :, i, :] = i / self.overlap
                            
                            if x > 0:
                                for i in range(min(self.overlap, x_end - x_start)):
                                    blend_x[:, :, :, i] = i / self.overlap
                            
                            blend = blend_y * blend_x
                            
                            # Apply blending
                            depth[b:b+1, :, y_start:y_end, x_start:x_end] = (
                                depth[b:b+1, :, y_start:y_end, x_start:x_end] * (1 - blend) + 
                                block * blend
                            )
                        else:
                            depth[b:b+1, :, y_start:y_end, x_start:x_end] = block
        
        return depth
    
    def fft_diffuse_block(self, block, cv, ch, l=0.24, steps=10):
        """
        Apply FFT-based diffusion to a block with uniform diffusion coefficients.
        This simulates multiple steps of diffusion in the frequency domain.
        """
        # Average diffusion coefficients for the block
        cv_mean = cv.mean()
        ch_mean = ch.mean()
        
        # Apply FFT
        fft_block = torch.fft.rfft2(block)
        
        # Create diffusion kernel in frequency domain
        h, w = block.shape[2], block.shape[3]
        ky = torch.arange(0, h).reshape(-1, 1).repeat(1, w//2 + 1).to(block.device) * (2 * np.pi / h)
        kx = torch.arange(0, w//2 + 1).reshape(1, -1).repeat(h, 1).to(block.device) * (2 * np.pi / w)
        
        # Diffusion operator in frequency domain
        # This simulates multiple steps of diffusion
        diffusion_operator = 1 - 2 * l * (cv_mean * (1 - torch.cos(ky)) + ch_mean * (1 - torch.cos(kx)))
        
        # Apply multiple steps of diffusion in frequency domain
        diffusion_operator = diffusion_operator ** steps
        
        # Apply the operator
        fft_block = fft_block * diffusion_operator.unsqueeze(0).unsqueeze(0)
        
        # Inverse FFT
        block = torch.fft.irfft2(fft_block, s=(h, w))
        
        return block

def adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8):
    # Implementation of the adjustment step. Eq (3) in paper.

    # Iss = subsample img
    img_ss = downsample(img)

    # Rss = source / Iss
    ratio_ss = source / (img_ss + eps)
    ratio_ss[mask_inv] = 1

    # R = NN upsample r
    ratio = upsample(ratio_ss)

    # img = img * R
    return img * ratio 

def _test_fft_gad():
    """
    Minimal test function to verify the FFT GAD implementation.
    """
    # Create a fake input
    guide = torch.rand(1, 3, 64, 64).cuda()
    source = torch.rand(1, 1, 16, 16).cuda()
    initial = F.interpolate(source, size=(64, 64), mode='bicubic').cuda()
    mask_lr = torch.ones(1, 1, 16, 16).cuda()

    sample = {
        'guide': guide,
        'source': source,
        'y_bicubic': initial,
        'mask_lr': mask_lr,
    }

    # Instantiate the model
    model = FFTGADBase(Npre=100, Ntrain=10).cuda()

    # Forward pass
    output = model(sample)
    print("Output keys:", output.keys())
    print("Predicted shape:", output['y_pred'].shape)

if __name__ == "__main__":
    _test_fft_gad() 
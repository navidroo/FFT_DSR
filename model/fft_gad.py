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
        print(f"Initializing FFTGADBase with Npre={Npre}, Ntrain={Ntrain}, block_size={block_size}, overlap={overlap}")

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
        self.block_size = block_size
        self.overlap = overlap
 
        if feature_extractor=='none': 
            # RGB verion of DADA does not need a deep feature extractor
            print("Using RGB version (no feature extractor)")
            self.feature_extractor = None
            self.Ntrain = 0
            self.logk = torch.log(torch.tensor(0.03))

        elif feature_extractor=='UNet':
            # Learned verion of DADA
            print("Using UNet feature extractor")
            self.feature_extractor = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bicubic'),
                smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=INPUT_DIM),
                torch.nn.AvgPool2d(kernel_size=2, stride=2)
            )
            self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))

        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')
             
    def forward(self, sample, train=False, deps=0.1):
        print("FFTGADBase forward pass started")
        guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']

        # assert that all values are positive, otherwise shift depth map to positives
        if source.min() <= deps:
            print("Warning: The forward function was called with negative depth values. Values were temporarly shifted. Consider using unnormalized depth values for stability.")
            source += deps
            sample['y_bicubic'] += deps
            shifted = True
        else:
            shifted = False

        print("Starting diffuse method")
        y_pred, aux = self.diffuse(sample['y_bicubic'].clone(), guide.clone(), source, mask_lr < 0.5,
                 K=torch.exp(self.logk), verbose=False, train=train)
        print("Diffuse method completed")

        # revert the shift
        if shifted:
            y_pred -= deps

        print("FFTGADBase forward pass completed")
        return {**{'y_pred': y_pred}, **aux}

    def diffuse(self, img, guide, source, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False):
        print("Diffuse method started")
        _,_,h,w = guide.shape
        _,_,sh,sw = source.shape

        # Define Downsampling operations that depend on the input size
        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')

        # Deep Learning version or RGB version to calculate the coefficients
        if self.feature_extractor is None: 
            print("Using RGB features")
            guide_feats = torch.cat([guide, img], 1) 
        else:
            print("Extracting deep features")
            guide_feats = self.feature_extractor(torch.cat([guide, img-img.mean((1,2,3), keepdim=True)], 1))
            print("Feature extraction completed")
        
        # Convert the features to coefficients with the Perona-Malik edge-detection function
        print("Computing diffusion coefficients")
        cv, ch = c(guide_feats, K=K)
        print("Diffusion coefficients computed")
        
        # Identify regions with relatively uniform diffusion coefficients
        # These are regions where FFT-based diffusion can be applied
        print("Identifying uniform regions")
        uniform_regions = self.identify_uniform_regions(cv, ch)
        print(f"Uniform regions identified: {uniform_regions.float().mean().item()*100:.2f}% of image")

        # Iterations without gradient
        if self.Npre > 0: 
            with torch.no_grad():
                Npre = randrange(self.Npre) if train else self.Npre
                print(f"Starting FFT diffusion with Npre={Npre}")
                
                # Apply FFT diffusion to uniform regions first
                print("Applying FFT diffusion")
                img = self.fft_diffuse(img, cv, ch, uniform_regions, l=l)
                print("FFT diffusion completed")
                
                # Then apply standard diffusion for remaining iterations
                print(f"Starting standard diffusion for {min(500, Npre)} iterations")
                for t in range(min(500, Npre)):
                    if t % 100 == 0:
                        print(f"Standard diffusion iteration {t}/{min(500, Npre)}")                     
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, eps=eps)
                print("Standard diffusion completed")

        # Iterations with gradient
        if self.Ntrain > 0: 
            print(f"Starting gradient-enabled diffusion for {self.Ntrain} iterations")
            for t in range(self.Ntrain):
                if t % 50 == 0:
                    print(f"Gradient diffusion iteration {t}/{self.Ntrain}")
                img = diffuse_step(cv, ch, img, l=l)
                img = adjust_step(img, source, mask_inv, upsample, downsample, eps=eps)
            print("Gradient-enabled diffusion completed")

        print("Diffuse method completed")
        return img, {"cv": cv, "ch": ch}
    
    def identify_uniform_regions(self, cv, ch, threshold=0.1):
        """
        Identify regions with relatively uniform diffusion coefficients.
        Returns a mask where 1 indicates uniform regions suitable for FFT.
        """
        print(f"CV shape: {cv.shape}, CH shape: {ch.shape}")
        
        # Calculate local variance of diffusion coefficients
        # Need to handle different shapes of cv and ch
        cv_var = F.avg_pool2d(cv**2, 3, stride=1, padding=1) - F.avg_pool2d(cv, 3, stride=1, padding=1)**2
        ch_var = F.avg_pool2d(ch**2, 3, stride=1, padding=1) - F.avg_pool2d(ch, 3, stride=1, padding=1)**2
        
        # Resize to match the smaller size
        _, _, h_cv, w_cv = cv_var.shape
        _, _, h_ch, w_ch = ch_var.shape
        h_min = min(h_cv, h_ch)
        w_min = min(w_cv, w_ch)
        
        if h_cv > h_min or w_cv > w_min:
            cv_var = cv_var[:, :, :h_min, :w_min]
        
        if h_ch > h_min or w_ch > w_min:
            ch_var = ch_var[:, :, :h_min, :w_min]
        
        # Regions with low variance are considered uniform
        uniform_regions = (cv_var < threshold) & (ch_var < threshold)
        
        print(f"Uniform regions shape: {uniform_regions.shape}")
        return uniform_regions
    
    def fft_diffuse(self, depth, cv, ch, uniform_regions, l=0.24, fft_steps=10):
        """
        Apply FFT-based diffusion to accelerate the process in uniform regions.
        """
        print(f"FFT diffuse started with fft_steps={fft_steps}")
        batch_size, channels, height, width = depth.shape
        print(f"Image shape: {batch_size}x{channels}x{height}x{width}")
        
        # Get the shapes of the diffusion coefficients
        _, _, h_cv, w_cv = cv.shape
        _, _, h_ch, w_ch = ch.shape
        
        # Process the image in blocks
        block_count = 0
        fft_applied_count = 0
        
        for b in range(batch_size):
            for y in range(0, height, self.block_size - self.overlap):
                for x in range(0, width, self.block_size - self.overlap):
                    block_count += 1
                    # Define block boundaries with overlap
                    y_end = min(y + self.block_size, height)
                    x_end = min(x + self.block_size, width)
                    y_start = max(0, y)
                    x_start = max(0, x)
                    
                    # Extract block
                    block = depth[b:b+1, :, y_start:y_end, x_start:x_end]
                    
                    # Handle edge cases for diffusion coefficients
                    cv_y_end = min(y_end-1, h_cv)
                    cv_x_end = min(x_end, w_cv)
                    ch_y_end = min(y_end, h_ch)
                    ch_x_end = min(x_end-1, w_ch)
                    
                    try:
                        # Make sure we don't go out of bounds
                        if y_start >= cv_y_end or x_start >= cv_x_end:
                            continue
                            
                        if y_start >= ch_y_end or x_start >= ch_x_end:
                            continue
                            
                        block_cv = cv[b:b+1, :, y_start:cv_y_end, x_start:cv_x_end]
                        block_ch = ch[b:b+1, :, y_start:ch_y_end, x_start:ch_x_end]
                        
                        # Get the corresponding region from uniform_regions
                        ur_y_end = min(y_end, uniform_regions.shape[2])
                        ur_x_end = min(x_end, uniform_regions.shape[3])
                        
                        if y_start >= ur_y_end or x_start >= ur_x_end:
                            continue
                            
                        block_uniform = uniform_regions[b:b+1, :, y_start:ur_y_end, x_start:ur_x_end]
                        
                        # If the block is mostly uniform, apply FFT diffusion
                        block_uniformity = block_uniform.float().mean().item()
                        if block_uniformity > 0.7:
                            fft_applied_count += 1
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
                    except Exception as e:
                        print(f"Error processing block at ({y_start}:{y_end}, {x_start}:{x_end}): {e}")
        
        print(f"FFT diffuse completed. Processed {block_count} blocks, applied FFT to {fft_applied_count} blocks ({fft_applied_count/block_count*100:.2f}%)")
        return depth
    
    def fft_diffuse_block(self, block, cv, ch, l=0.24, steps=10):
        """
        Apply FFT-based diffusion to a block with uniform diffusion coefficients.
        This simulates multiple steps of diffusion in the frequency domain.
        """
        # Average diffusion coefficients for the block
        try:
            cv_mean = cv.mean()
            ch_mean = ch.mean()
            
            print(f"Block shape: {block.shape}, CV shape: {cv.shape}, CH shape: {ch.shape}")
            print(f"CV mean: {cv_mean.item()}, CH mean: {ch_mean.item()}")
            
            # Apply FFT
            print("Applying FFT...")
            fft_block = torch.fft.rfft2(block)
            print(f"FFT block shape: {fft_block.shape}")
            
            # Create diffusion kernel in frequency domain
            h, w = block.shape[2], block.shape[3]
            print(f"Creating frequency domain kernel with h={h}, w={w}")
            ky = torch.arange(0, h).reshape(-1, 1).repeat(1, w//2 + 1).to(block.device) * (2 * np.pi / h)
            kx = torch.arange(0, w//2 + 1).reshape(1, -1).repeat(h, 1).to(block.device) * (2 * np.pi / w)
            
            # Diffusion operator in frequency domain
            # This simulates multiple steps of diffusion
            print("Creating diffusion operator...")
            diffusion_operator = 1 - 2 * l * (cv_mean * (1 - torch.cos(ky)) + ch_mean * (1 - torch.cos(kx)))
            print(f"Diffusion operator shape: {diffusion_operator.shape}")
            
            # Apply multiple steps of diffusion in frequency domain
            print(f"Applying {steps} steps of diffusion...")
            diffusion_operator = diffusion_operator ** steps
            
            # Apply the operator
            print("Applying operator to FFT block...")
            fft_block = fft_block * diffusion_operator.unsqueeze(0).unsqueeze(0)
            
            # Inverse FFT
            print("Applying inverse FFT...")
            block = torch.fft.irfft2(fft_block, s=(h, w))
            print(f"Result block shape: {block.shape}")
            
            return block
        except Exception as e:
            import traceback
            print(f"Error in FFT diffusion block: {e}")
            traceback.print_exc()
            # Return original block if FFT fails
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
# FFT-based Diffusion for Accelerated Guided Depth Super-Resolution
# Based on the original implementation by Nando Metzger

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FFTDiffuseBase(nn.Module):
    def __init__(
            self, 
            Niter=2000,  # Reduced number of iterations due to FFT acceleration
            block_size=64,  # Size of blocks for block-wise FFT processing
            overlap=16,    # Overlap between blocks
    ):
        super().__init__()
        print(f"Initializing FFTDiffuseBase with Niter={Niter}, block_size={block_size}, overlap={overlap}")
        self.Niter = Niter
        self.logk = torch.log(torch.tensor(0.03))
        self.block_size = block_size
        self.overlap = overlap
        
    def forward(self, sample, train=False, deps=0.1):
        print("FFTDiffuseBase forward pass started")
        guide, initial = sample['guide'], sample['initial']

        # assert that all values are positive, otherwise shift depth map to positives
        if initial.min() <= deps:
            print("Warning: The forward function was called with negative depth values. Values were temporarly shifted. Consider using unnormalized depth values for stability.")
            initial += deps
            shifted = True
        else:
            shifted = False

        # Execute diffusion
        print("Starting diffuse method")
        y_pred, aux = self.diffuse(initial.clone(), guide.clone(), K=torch.exp(self.logk))
        print("Diffuse method completed")

        # revert the shift
        if shifted:
            y_pred -= deps

        print("FFTDiffuseBase forward pass completed")
        return {**{'y_pred': y_pred}, **aux}

    def diffuse(self, depth, guide, l=0.24, K=0.01):
        print("Diffuse method started")
        _, _, h, w = guide.shape
        
        # Convert the features to coefficients with the Perona-Malik edge-detection function
        print("Computing diffusion coefficients")
        cv, ch = c(guide, K=K)
        print("Diffusion coefficients computed")
        
        # Identify regions with relatively uniform diffusion coefficients
        # These are regions where FFT-based diffusion can be applied
        print("Identifying uniform regions")
        uniform_regions = self.identify_uniform_regions(cv, ch)
        print(f"Uniform regions identified: {uniform_regions.float().mean().item()*100:.2f}% of image")
        
        # Perform block-wise FFT diffusion
        print("Starting FFT diffusion")
        depth = self.fft_diffuse(depth, cv, ch, uniform_regions, l=l)
        print("FFT diffusion completed")
        
        # Perform standard diffusion for remaining iterations or non-uniform regions
        print(f"Starting standard diffusion for {min(1000, self.Niter)} iterations")
        for t in range(min(1000, self.Niter)):
            if t % 100 == 0:
                print(f"Standard diffusion iteration {t}/{min(1000, self.Niter)}")
            depth = diffuse_step(cv, ch, depth, l=l)
        print("Standard diffusion completed")

        print("Diffuse method completed")
        return depth, {"cv": cv, "ch": ch}
    
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
        cv_mean = cv.mean()
        ch_mean = ch.mean()
        
        try:
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
        except Exception as e:
            print(f"Error in FFT diffusion block: {e}")
            # Return original block if FFT fails
            return block

@torch.jit.script
def diffuse_step(cv, ch, I, l: float=0.24):
    # Anisotropic Diffusion implementation, Eq. (1) in paper.

    # calculate diffusion update as increments
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    tv = l * cv * dv # vertical transmissions
    I[:,:,1:,:] -= tv
    I[:,:,:-1,:] += tv 

    th = l * ch * dh # horizontal transmissions
    I[:,:,:,1:] -= th
    I[:,:,:,:-1] += th 
    
    return I
  
def c(I, K: float=0.03):
    # apply function to both dimensions
    cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]), 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]), 1), 1), K)
    return cv, ch

def g(x, K: float=0.03):
    # Perona-Malik edge detection
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))

def _test_fft_diffuse():
    """
    Minimal test function to verify the FFT diffusion step.
    """
    # Create a fake input (batch=1, channel=1/3, height=64, width=64)
    guide = torch.rand(1, 3, 64, 64)
    initial = torch.rand(1, 1, 64, 64) + 1.0

    sample = {
        'guide': guide,
        'initial': initial,
    }

    # Instantiate the model
    model = FFTDiffuseBase(Niter=100)

    # Forward pass
    output = model(sample)
    print("Output keys:", output.keys())
    print("Predicted shape:", output['y_pred'].shape)

if __name__ == "__main__":
    _test_fft_diffuse() 
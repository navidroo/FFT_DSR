import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
from torchdiffeq import odeint_adjoint as odeint

from model.fft_gad import FFTGADBase, diffuse_step, adjust_step
from model.fft_diffusion import c, g

INPUT_DIM = 4
FEATURE_DIM = 64

class MultiResODEGAD(FFTGADBase):
    """
    Multi-Resolution ODE-based Guided Anisotropic Diffusion for improved performance at high scaling factors.
    Extends the FFTGADBase model with frequency band decomposition and ODE-based diffusion.
    """
    
    def __init__(
            self, feature_extractor='UNet',
            Npre=2000, Ntrain=256,
            block_size=64, overlap=16,
            adaptive_block_size=False, scaling_factor=8,
            bands=3, # Number of frequency bands
            ode_rtol=1e-3, ode_atol=1e-3, # ODE solver tolerances
            ode_method='dopri5' # ODE solver method
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            Npre=Npre, 
            Ntrain=Ntrain,
            block_size=block_size,
            overlap=overlap,
            adaptive_block_size=adaptive_block_size,
            scaling_factor=scaling_factor
        )
        
        # Multi-resolution parameters
        self.bands = bands
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self.ode_method = ode_method
        
        # Create separate feature extractors for different frequency bands
        if self.feature_extractor_name == 'UNet':
            print(f"Creating {bands} frequency-specific feature extractors")
            self.band_feature_extractors = nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bicubic'),
                    smp.Unet('resnet34', classes=FEATURE_DIM, in_channels=INPUT_DIM),
                    torch.nn.AvgPool2d(kernel_size=2, stride=2)
                ) for _ in range(bands-1)  # High frequency bands
            ])
            
            # Learnable band mixing parameters
            self.band_weights = nn.Parameter(torch.ones(bands)/bands)
            
        print(f"Initialized MultiResODEGAD with {bands} frequency bands")
        
    def forward(self, sample, train=False, deps=0.1):
        print("MultiResODEGAD forward pass started")
        guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']

        # Assert that all values are positive, otherwise shift depth map to positives
        if source.min() <= deps:
            print("Warning: The forward function was called with negative depth values. Values were temporarly shifted. Consider using unnormalized depth values for stability.")
            source += deps
            sample['y_bicubic'] += deps
            shifted = True
        else:
            shifted = False

        print("Starting multi-resolution diffuse method")
        y_pred, aux = self.diffuse(sample['y_bicubic'].clone(), guide.clone(), source, mask_lr < 0.5,
                 K=torch.exp(self.logk), verbose=False, train=train)
        print("Multi-resolution diffuse method completed")

        # Revert the shift
        if shifted:
            y_pred -= deps

        print("MultiResODEGAD forward pass completed")
        return {**{'y_pred': y_pred}, **aux}
        
    def diffuse(self, img, guide, source, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False):
        """
        Multi-resolution diffusion process using ODE solver
        """
        print("Multi-resolution diffuse method started")
        _,_,h,w = guide.shape
        _,_,sh,sw = source.shape

        # Define downsampling/upsampling operations
        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')

        # Decompose input into frequency bands
        print(f"Decomposing input into {self.bands} frequency bands")
        bands = self.decompose_to_frequency_bands(img, self.bands)
        
        # Process each band differently
        processed_bands = []
        band_features = []
        
        # Extract guide features for the main (base) band
        if self.feature_extractor is not None:
            print("Extracting main band features")
            main_guide_feats = self.feature_extractor(
                torch.cat([guide, img-img.mean((1,2,3), keepdim=True)], 1)
            )
            band_features.append(main_guide_feats)
        else:
            main_guide_feats = torch.cat([guide, img], 1)
            band_features.append(main_guide_feats)
            
        # Extract features for other bands if available
        if hasattr(self, 'band_feature_extractors'):
            for i, band_img in enumerate(bands[1:]):
                # Combine this band with guide
                if verbose:
                    print(f"Extracting features for band {i+1}")
                
                # Up-interpolate band to match guide size if needed
                if band_img.shape[2:] != guide.shape[2:]:
                    band_img = F.interpolate(band_img, guide.shape[2:], mode='bilinear')
                    
                band_guide_feats = self.band_feature_extractors[i](
                    torch.cat([guide, band_img-band_img.mean((1,2,3), keepdim=True)], 1)
                )
                band_features.append(band_guide_feats)
                
        # Process main band (low frequencies) - can use standard diffusion
        if verbose:
            print("Processing main frequency band")
        
        # Generate diffusion coefficients for the main band
        cv_main, ch_main = c(band_features[0], K=K)
        
        # ODE function for diffusion process
        def diffusion_ode_func(t, y):
            # Basic diffusion update
            dy = diffuse_step(cv_main, ch_main, y.clone(), l=l) - y
            
            # Adjustment step (continuous version)
            y_downsampled = downsample(y)
            ratio_ss = source / (y_downsampled + eps)
            ratio_ss[mask_inv] = 1
            ratio = upsample(ratio_ss)
            
            # Apply adjustment with t-dependent weight (more at start, less at end)
            adjustment = y * (ratio - 1.0)
            adjustment_weight = torch.sigmoid(10 * (1 - t))
            
            return dy + adjustment_weight * adjustment
        
        # Process main band through ODE
        if train and self.Ntrain > 0:
            t_span = torch.linspace(0, 1, 10).to(img.device)
            bands[0] = odeint(
                diffusion_ode_func,
                bands[0],
                t_span,
                method=self.ode_method,
                rtol=self.ode_rtol,
                atol=self.ode_atol
            )[-1]  # Take final state
        else:
            # For evaluation or if Ntrain=0, use standard diffusion
            bands[0] = self.standard_diffusion(bands[0], cv_main, ch_main, source, mask_inv, 
                                            downsample, upsample, l, eps, train)
        
        processed_bands.append(bands[0])
        
        # Process higher frequency bands
        for i in range(1, len(bands)):
            if i < len(band_features):
                # We have specific features for this band
                band_feats = band_features[i]
            else:
                # Fallback to main features
                band_feats = band_features[0]
                
            if verbose:
                print(f"Processing frequency band {i}")
                
            # Generate diffusion coefficients for this band
            cv_band, ch_band = c(band_feats, K=K)
            
            # For higher bands, we use specialized ODE with edge preservation
            def high_freq_ode_func(t, y):
                # Edge-preserving diffusion (reduce diffusion near edges)
                edges = torch.mean(torch.abs(band_feats), dim=1, keepdim=True)
                edge_weight = torch.exp(-edges * 5.0)  # Reduce diffusion at edges
                
                # Modified diffusion coefficients to preserve edges
                cv_mod = cv_band * edge_weight
                ch_mod = ch_band * edge_weight
                
                # Compute diffusion update with edge preservation
                return diffuse_step(cv_mod, ch_mod, y.clone(), l=l*0.5) - y
            
            # Process through ODE solver (for high-frequency, we always use ODE)
            t_span = torch.linspace(0, 1, 5).to(img.device)  # Fewer steps for high-freq
            bands[i] = odeint(
                high_freq_ode_func,
                bands[i],
                t_span,
                method=self.ode_method,
                rtol=self.ode_rtol*10,  # More relaxed tolerance for high-freq
                atol=self.ode_atol*10
            )[-1]  # Take final state
            
            processed_bands.append(bands[i])
            
        # Recombine bands using learned weights if training
        if train and hasattr(self, 'band_weights'):
            band_weights = F.softmax(self.band_weights, dim=0)
            if verbose:
                print(f"Recombining bands with weights: {band_weights.detach().cpu().numpy()}")
        else:
            # Equal weights during evaluation or if no learned weights
            band_weights = torch.ones(len(processed_bands)).to(img.device) / len(processed_bands)
            
        # Recombine processed bands
        result = self.recombine_frequency_bands(processed_bands, band_weights)
        
        # Final adjustment step to ensure consistency with low-res input
        result = adjust_step(result, source, mask_inv, upsample, downsample, eps=eps)
        
        print("Multi-resolution diffuse method completed")
        return result, {"cv": cv_main, "ch": ch_main}
        
    def standard_diffusion(self, img, cv, ch, source, mask_inv, downsample, upsample, l=0.24, eps=1e-8, train=False):
        """Standard diffusion process for the base frequency band"""
        # Iterations without gradient
        if self.Npre > 0: 
            with torch.no_grad():
                Npre = min(2000, self.Npre) if train else self.Npre
                for t in range(Npre):
                    if t % 200 == 0:
                        print(f"Diffusion iteration {t}/{Npre}")                     
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, eps=eps)

        # Iterations with gradient
        if self.Ntrain > 0: 
            for t in range(self.Ntrain):
                if t % 50 == 0:
                    print(f"Gradient diffusion iteration {t}/{self.Ntrain}")
                img = diffuse_step(cv, ch, img, l=l)
                img = adjust_step(img, source, mask_inv, upsample, downsample, eps=eps)

        return img
        
    def decompose_to_frequency_bands(self, img, num_bands=3):
        """
        Decompose image into frequency bands using Laplacian pyramid-like approach
        Returns: list of tensors, from low to high frequency
        """
        bands = []
        current = img.clone()
        
        # Create n-1 high-frequency bands
        for i in range(num_bands - 1):
            # Downsample then upsample to get low-frequency version
            downsampled = F.avg_pool2d(current, kernel_size=2, stride=2)
            upsampled = F.interpolate(downsampled, current.shape[2:], mode='bilinear', align_corners=False)
            
            # High frequency is the residual
            high_freq = current - upsampled
            
            # Store low frequency for next iteration
            current = downsampled
            
            # Store high frequency band (residual)
            bands.append(high_freq)
        
        # The final band is the lowest frequency
        bands.append(current)
        
        # Reverse list so low frequencies come first
        return bands[::-1]
        
    def recombine_frequency_bands(self, bands, weights=None):
        """
        Recombine frequency bands into a single image
        bands: list of tensors from low to high frequency
        weights: optional weights for each band, defaults to equal weighting
        """
        if weights is None:
            weights = torch.ones(len(bands)).to(bands[0].device) / len(bands)
            
        # First, ensure all bands are at the same resolution as the first band
        target_size = bands[0].shape[2:]
        aligned_bands = []
        
        for i, band in enumerate(bands):
            if band.shape[2:] != target_size:
                band = F.interpolate(band, target_size, mode='bilinear', align_corners=False)
            aligned_bands.append(band)
            
        # Now combine the bands
        result = torch.zeros_like(aligned_bands[0])
        for i, band in enumerate(aligned_bands):
            result = result + weights[i] * band
            
        return result
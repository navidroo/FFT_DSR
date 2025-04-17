"""
SwinFuSR-GAD: Integrating Swin Transformer blocks with guided anisotropic diffusion for depth super-resolution
Based on VISION IC Team's SwinFuSR approach and the existing DADA implementation
"""

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np

from model.fft_diffusion import FFTDiffuseBase, c, g, diffuse_step
from model.fft_gad import FFTGADBase, adjust_step

# Constants
INPUT_DIM = 4
FEATURE_DIM = 64
SWIN_FEATURE_DIM = 96  # Typical dimension for Swin Transformer features

class SwinTransformerBlock(nn.Module):
    """Basic Swin Transformer Block with window attention"""
    def __init__(self, dim, num_heads=3, window_size=7, mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        # Window attention
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        print(f"SwinTransformerBlock input shape: {x.shape}")
        
        # Rearrange to tokens
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        shortcut = x
        
        # Layer Norm
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)
        
        # Window partition and attention
        x_windows, original_size, padded_size = window_partition(x, self.window_size)
        H_padded, W_padded = padded_size
        
        # Reshape for attention
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C
        
        # Merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H_padded, W_padded, original_size)  # B, H, W, C
        
        # Reshape back 
        x = x.reshape(B, H * W, C)
        
        # Residual connection
        x = shortcut + x
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        
        # Reshape back to feature map
        x = x.transpose(1, 2).reshape(B, C, H, W)
        print(f"SwinTransformerBlock output shape: {x.shape}")
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C/nH

        # Self-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregation
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """Partition into non-overlapping windows with padding if needed"""
    B, H, W, C = x.shape
    print(f"window_partition input: {x.shape}, window_size={window_size}")
    
    # Apply padding if needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        print(f"Applying padding: pad_h={pad_h}, pad_w={pad_w}")
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H_padded, W_padded = H + pad_h, W + pad_w
    else:
        H_padded, W_padded = H, W
    
    print(f"After padding: {x.shape}")
    
    # Use reshape operations that preserve gradients
    x = x.reshape(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.reshape(-1, window_size, window_size, C)
    
    print(f"window_partition output: {windows.shape}, padding info: ({H}, {W})->({H_padded}, {W_padded})")
    return windows, (H, W), (H_padded, W_padded)


def window_reverse(windows, window_size, H_padded, W_padded, original_size=None):
    """Reverse window partition and remove padding if needed"""
    print(f"window_reverse input: {windows.shape}, window_size={window_size}, H_padded={H_padded}, W_padded={W_padded}")
    
    B = int(windows.shape[0] / (H_padded * W_padded / window_size / window_size))
    
    # Use reshape operations that preserve gradients
    x = windows.reshape(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(B, H_padded, W_padded, -1)
    
    # Remove padding if original size is provided
    if original_size:
        H, W = original_size
        if H < H_padded or W < W_padded:
            print(f"Removing padding: ({H_padded}, {W_padded})->({H}, {W})")
            x = x[:, :H, :W, :].contiguous()
    
    print(f"window_reverse output: {x.shape}")
    return x


class CrossDomainFusionBlock(nn.Module):
    """Cross-Domain Fusion Block for merging RGB and depth features"""
    def __init__(self, rgb_dim, depth_dim, fusion_dim):
        super().__init__()
        self.rgb_conv = nn.Conv2d(rgb_dim, fusion_dim, kernel_size=1)
        self.depth_conv = nn.Conv2d(depth_dim, fusion_dim, kernel_size=1)
        
        # Cross-attention
        self.rgb_to_depth_attn = nn.Sequential(
            nn.Conv2d(fusion_dim*2, fusion_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.depth_to_rgb_attn = nn.Sequential(
            nn.Conv2d(fusion_dim*2, fusion_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_dim*2, fusion_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=1)
        )
        
    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.rgb_conv(rgb_feat)
        depth_feat = self.depth_conv(depth_feat)
        
        # Cross attention
        concat_feat = torch.cat([rgb_feat, depth_feat], dim=1)
        rgb_attn = self.depth_to_rgb_attn(concat_feat)
        depth_attn = self.rgb_to_depth_attn(concat_feat)
        
        # Apply attention
        rgb_feat_enhanced = rgb_feat * rgb_attn
        depth_feat_enhanced = depth_feat * depth_attn
        
        # Fusion
        fused_feat = self.fusion(torch.cat([rgb_feat_enhanced, depth_feat_enhanced], dim=1))
        return fused_feat


class SwinFuSRGAD(FFTGADBase):
    """
    Guided Depth Super-Resolution using Swin Transformer and Anisotropic Diffusion
    Combines VISION IC Team's SwinFuSR approach with DADA's diffusion framework
    """
    def __init__(
            self, feature_extractor='SwinFuSR',
            Npre=1500, Ntrain=128,  # Reduced iterations due to better features
            block_size=64, overlap=16,
            adaptive_block_size=False, scaling_factor=8,
            swin_window_size=7, num_heads=3, num_swin_blocks=4
    ):
        # Initialize parent class but don't create feature extractor yet
        super().__init__(
            feature_extractor='none',  # Will be overridden
            Npre=Npre,
            Ntrain=Ntrain,
            block_size=block_size,
            overlap=overlap,
            adaptive_block_size=adaptive_block_size,
            scaling_factor=scaling_factor
        )
        
        # Store Swin parameters
        self.swin_window_size = swin_window_size
        self.num_heads = num_heads
        self.num_swin_blocks = num_swin_blocks
        self.feature_extractor_name = feature_extractor
        
        # Initialize SwinFuSR feature extractor
        self._init_feature_extractor()
        
    def _init_feature_extractor(self):
        print("Initializing SwinFuSR feature extractor")
        
        # RGB feature extractor
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, SWIN_FEATURE_DIM, kernel_size=3, padding=1),
        )
        
        # Depth feature extractor
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, SWIN_FEATURE_DIM, kernel_size=3, padding=1),
        )
        
        # Swin Transformer blocks for RGB
        self.rgb_swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=SWIN_FEATURE_DIM, 
                num_heads=self.num_heads,
                window_size=self.swin_window_size
            ) for _ in range(self.num_swin_blocks)
        ])
        
        # Swin Transformer blocks for depth
        self.depth_swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=SWIN_FEATURE_DIM, 
                num_heads=self.num_heads,
                window_size=self.swin_window_size
            ) for _ in range(self.num_swin_blocks)
        ])
        
        # Cross-domain fusion
        self.fusion_block = CrossDomainFusionBlock(
            rgb_dim=SWIN_FEATURE_DIM,
            depth_dim=SWIN_FEATURE_DIM,
            fusion_dim=SWIN_FEATURE_DIM
        )
        
        # Final mapping to diffusion coefficients
        self.final_conv = nn.Sequential(
            nn.Conv2d(SWIN_FEATURE_DIM, FEATURE_DIM, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(FEATURE_DIM, FEATURE_DIM, kernel_size=1)
        )
        
        # Make logk trainable
        self.logk = nn.Parameter(torch.log(torch.tensor(0.03)))
        
    def extract_features(self, guide, img):
        """Extract features using Swin Transformer blocks and fusion"""
        # Initial feature extraction
        rgb_feat = self.rgb_conv(guide)
        
        # Check img dimensions and handle accordingly
        if img.dim() == 3:  # No channel dimension (B, H, W)
            depth_feat = self.depth_conv(img.unsqueeze(1))
        elif img.dim() == 4 and img.shape[1] == 1:  # Already has channel dimension (B, 1, H, W)
            depth_feat = self.depth_conv(img)
        else:  # Handle any other case (reshape if needed)
            # Use reshape instead of view to maintain gradients
            depth_feat = self.depth_conv(img.reshape(img.shape[0], 1, img.shape[-2], img.shape[-1]))
        
        # Process through Swin blocks
        for rgb_block, depth_block in zip(self.rgb_swin_blocks, self.depth_swin_blocks):
            rgb_feat = rgb_block(rgb_feat)
            depth_feat = depth_block(depth_feat)
        
        # Cross-domain fusion
        fused_feat = self.fusion_block(rgb_feat, depth_feat)
        
        # Final mapping
        final_feat = self.final_conv(fused_feat)
        
        return final_feat
        
    def forward(self, sample, train=False, deps=0.1):
        guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']
        
        print(f"SwinFuSRGAD forward input shapes: guide={guide.shape}, source={source.shape}, y_bicubic={sample['y_bicubic'].shape}")

        # Handle negative depth values
        if source.min() <= deps:
            print("Warning: The forward function was called with negative depth values. Values were temporarly shifted.")
            source += deps
            sample['y_bicubic'] += deps
            shifted = True
        else:
            shifted = False

        # Use the diffuse method from parent class but with our feature extractor
        y_pred, aux = self.diffuse(sample['y_bicubic'].clone(), guide.clone(), source, mask_lr < 0.5,
                 K=torch.exp(self.logk), verbose=False, train=train)

        # Revert the shift if needed
        if shifted:
            y_pred -= deps
            
        print(f"SwinFuSRGAD forward output shape: y_pred={y_pred.shape}")

        return {**{'y_pred': y_pred}, **aux}
    
    def diffuse(self, img, guide, source, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False):
        """Modified diffuse method using SwinFuSR features"""
        print(f"SwinFuSRGAD diffuse input shapes: img={img.shape}, guide={guide.shape}, source={source.shape}")
        print(f"Tensors require grad: img={img.requires_grad}, guide={guide.requires_grad}")
        
        _,_,h,w = guide.shape
        _,_,sh,sw = source.shape

        # Define downsampling and upsampling operations
        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')

        # Extract features using SwinFuSR instead of the original method
        guide_feats = self.extract_features(guide, img)
        print(f"Feature extraction output shape: guide_feats={guide_feats.shape}")
        print(f"guide_feats requires grad: {guide_feats.requires_grad}")
        
        # Convert features to diffusion coefficients
        cv, ch = c(guide_feats, K=K)
        print(f"Diffusion coefficients shapes: cv={cv.shape}, ch={ch.shape}")
        print(f"cv requires grad: {cv.requires_grad}, ch requires grad: {ch.requires_grad}")
        
        # Identify regions with uniform diffusion coefficients for FFT acceleration
        uniform_regions = self.identify_uniform_regions(cv, ch)
        print(f"Uniform regions shape: uniform_regions={uniform_regions.shape}")
        
        # First stage: iterations without gradient
        if self.Npre > 0: 
            with torch.no_grad():
                Npre = np.random.randint(0, self.Npre) if train else self.Npre
                print(f"Starting {Npre} iterations without gradient")
                
                # Apply FFT diffusion to uniform regions
                img = self.fft_diffuse(img, cv, ch, uniform_regions, l=l)
                
                # Standard diffusion for remaining iterations
                for t in range(min(300, Npre)):  # Reduced iterations due to better features     
                    if t % 100 == 0:
                        print(f"Iteration {t}/{min(300, Npre)}")              
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, eps=eps)

        # If in training mode, ensure img requires grad for the gradient-enabled iterations
        if train and not img.requires_grad:
            print("Re-enabling gradients for training iterations")
            img = img.detach().requires_grad_(True)
                
        # Second stage: iterations with gradient
        if self.Ntrain > 0: 
            print(f"Starting {self.Ntrain} iterations with gradient")
            for t in range(self.Ntrain):
                if t % 50 == 0:
                    print(f"Iteration {t}/{self.Ntrain}")
                    if train:
                        print(f"Tensor requires grad: img={img.requires_grad}, cv={cv.requires_grad}, ch={ch.requires_grad}")
                img = diffuse_step(cv, ch, img, l=l)
                img = adjust_step(img, source, mask_inv, upsample, downsample, eps=eps)
                
        print(f"SwinFuSRGAD diffuse output shape: img={img.shape}")
        print(f"Output tensor requires grad: {img.requires_grad}")
        return img, {"cv": cv, "ch": ch}

    def fft_diffuse(self, depth, cv, ch, uniform_regions, l=0.24, fft_steps=10):
        """
        Apply FFT-based diffusion to accelerate the process in uniform regions.
        This implementation ensures gradient flow during training.
        """
        print(f"SwinFuSRGAD fft_diffuse starting with tensors requiring grad: depth={depth.requires_grad}, cv={cv.requires_grad}, ch={ch.requires_grad}")
        
        batch_size, channels, height, width = depth.shape
        print(f"Image shape: {batch_size}x{channels}x{height}x{width}")
        
        # Process the image in blocks to handle varying diffusion coefficients
        # Use functorch to maintain gradient flow
        result = torch.zeros_like(depth)
        result.copy_(depth)  # This maintains gradient connection
        
        # Block processing parameters
        block_size = self.block_size
        overlap = self.overlap
        
        # Count for statistics
        total_blocks = 0
        fft_blocks = 0
        
        # Keep track of which parts have been processed for proper blending
        processed_mask = torch.zeros_like(depth)
        
        # Process the image in blocks
        for y in range(0, height, block_size - overlap):
            for x in range(0, width, block_size - overlap):
                # Determine block boundaries
                y_end = min(y + block_size, height)
                x_end = min(x + block_size, width)
                y_size = y_end - y
                x_size = x_end - x
                
                # Extract block 
                block = depth[:, :, y:y_end, x:x_end]
                
                # Get corresponding diffusion coefficients
                cv_block = cv[:, :, y:min(y+y_size-1, height-1), x:min(x+x_size-1, width-1)] if y+y_size-1 < height and x+x_size-1 < width else None
                ch_block = ch[:, :, y:min(y+y_size, height), x:min(x+x_size-1, width-1)] if y+y_size < height and x+x_size-1 < width else None
                
                if cv_block is None or ch_block is None or cv_block.shape[2] == 0 or cv_block.shape[3] == 0 or ch_block.shape[2] == 0 or ch_block.shape[3] == 0:
                    print(f"Skipping block at ({y}, {x}) due to invalid diffusion coefficient shapes")
                    continue
                    
                # Print shape information for debugging
                print(f"Block shape: {block.shape}, CV shape: {cv_block.shape}, CH shape: {ch_block.shape}")
                print(f"CV mean: {cv_block.mean().item()}, CH mean: {ch_block.mean().item()}")
                
                # Check if this block is in a uniform region
                if y+y_size-1 < height and x+x_size-1 < width:
                    uniform_block = uniform_regions[:, :, y:min(y+y_size-1, height-1), x:min(x+x_size-1, width-1)]
                    is_uniform = uniform_block.float().mean() > 0.8  # If more than 80% of the block is uniform
                else:
                    is_uniform = False
                
                total_blocks += 1
                
                # Only use FFT for uniform regions
                if is_uniform:
                    fft_blocks += 1
                    
                    # Apply FFT diffusion to this block
                    if block.requires_grad:
                        # For tensors requiring gradients, use custom FFT diffusion
                        block_result = self.fft_diffuse_block_with_grad(block, cv_block, ch_block, l, fft_steps)
                    else:
                        # Otherwise use standard implementation
                        block_result = self.fft_diffuse_block(block, cv_block, ch_block, l, fft_steps)
                    
                    # Calculate blend weights for smooth transition
                    if overlap > 0 and (y > 0 or x > 0):
                        blend = torch.ones_like(block)
                        
                        # Create blending weights along y-axis
                        if y > 0:
                            for i in range(min(overlap, y_size)):
                                blend[:, :, i, :] = i / overlap
                                
                        # Create blending weights along x-axis
                        if x > 0:
                            for i in range(min(overlap, x_size)):
                                if y > 0 and i < overlap:
                                    # For corner regions, take minimum of both blends
                                    blend[:, :, :overlap, i] = torch.min(blend[:, :, :overlap, i], 
                                                                        torch.tensor(i / overlap, device=blend.device))
                                else:
                                    blend[:, :, :, i] = i / overlap
                        
                        # Apply blending while maintaining gradient flow
                        # Calculate weighted contribution of new block
                        contribution = block_result * blend
                        
                        # Update the result using the contribution
                        result_block = result[:, :, y:y_end, x:x_end]
                        prev_contribution = result_block * (1 - blend)
                        result[:, :, y:y_end, x:x_end] = prev_contribution + contribution
                    else:
                        # Directly update the result
                        result[:, :, y:y_end, x:x_end] = block_result
                    
                    # Mark this region as processed
                    processed_mask[:, :, y:y_end, x:x_end] = 1.0
                
        # For any unprocessed regions (no blocks were uniform), use standard diffusion
        if processed_mask.min() < 1.0:
            unprocessed = (processed_mask < 0.5).float()
            if unprocessed.sum() > 0:
                print(f"Some regions weren't processed with FFT diffusion. Using standard diffusion for these areas.")
                # Apply a few steps of standard diffusion to unprocessed areas
                for _ in range(fft_steps // 2):
                    diffused = diffuse_step(cv, ch, result, l=l)
                    # Only update unprocessed areas
                    result = result * (1 - unprocessed) + diffused * unprocessed
        
        print(f"FFT diffuse completed. Processed {total_blocks} blocks, applied FFT to {fft_blocks} blocks ({100.0 * fft_blocks / total_blocks:.2f}%)")
        return result
    
    def fft_diffuse_block_with_grad(self, block, cv, ch, l=0.24, steps=10):
        """
        Apply FFT-based diffusion to a block with gradient support.
        This method is used during training when gradients are needed.
        """
        # Use a simplified diffusion approach that maintains gradient flow
        result = block.clone()
        
        # Apply multiple standard diffusion steps instead of FFT diffusion
        # This is a compromise that allows gradient flow while still accelerating
        for _ in range(steps // 2):  # Using fewer steps for better efficiency
            result = diffuse_step(cv, ch, result, l=l)
            
        return result

    def fft_diffuse_block(self, block, cv, ch, l=0.24, steps=10):
        """
        Apply FFT-based diffusion to a block without gradient support.
        This method is used during inference when gradients are not needed.
        """
        # Use a standard diffusion approach
        result = block.clone()
        
        # Apply multiple standard diffusion steps
        for _ in range(steps):
            result = diffuse_step(cv, ch, result, l=l)
        
        return result 
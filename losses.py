import torch
import torch.nn.functional as F

def get_loss(output, sample):
    y_pred = output['y_pred']
    y, mask_hr, mask_lr = (sample[k] for k in ('y', 'mask_hr', 'mask_lr'))

    l1_loss = l1_loss_func(y_pred, y, mask_hr)
    mse_loss = mse_loss_func(y_pred, y, mask_hr)

    loss = l1_loss*10

    return loss, {
        'l1_loss': l1_loss.detach().item(),
        'mse_loss': mse_loss.detach().item(),
        'optimization_loss': loss.detach().item(),
    }
        

def mse_loss_func(pred, gt, mask):
    """
    Compute MSE loss between prediction and ground truth, masked by the given mask.
    Resizes all tensors to match the prediction size before applying the mask.
    """
    # Get dimensions
    _, _, pred_h, pred_w = pred.shape
    _, _, mask_h, mask_w = mask.shape
    _, _, gt_h, gt_w = gt.shape
    
    # Resize all tensors to match prediction size
    if (mask_h != pred_h) or (mask_w != pred_w):
        mask = F.interpolate(mask, size=(pred_h, pred_w), mode='nearest')
    
    if (gt_h != pred_h) or (gt_w != pred_w):
        gt = F.interpolate(gt, size=(pred_h, pred_w), mode='bicubic', align_corners=True)
    
    # Apply mask and compute loss
    masked_pred = pred[mask == 1.]
    masked_gt = gt[mask == 1.]
    
    if masked_pred.numel() == 0 or masked_gt.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    return F.mse_loss(masked_pred, masked_gt)


def l1_loss_func(pred, gt, mask):
    """
    Compute L1 loss between prediction and ground truth, masked by the given mask.
    Resizes all tensors to match the prediction size before applying the mask.
    """
    # Get dimensions
    _, _, pred_h, pred_w = pred.shape
    _, _, mask_h, mask_w = mask.shape
    _, _, gt_h, gt_w = gt.shape
    
    # Resize all tensors to match prediction size
    if (mask_h != pred_h) or (mask_w != pred_w):
        mask = F.interpolate(mask, size=(pred_h, pred_w), mode='nearest')
    
    if (gt_h != pred_h) or (gt_w != pred_w):
        gt = F.interpolate(gt, size=(pred_h, pred_w), mode='bicubic', align_corners=True)
    
    # Apply mask and compute loss
    masked_pred = pred[mask == 1.]
    masked_gt = gt[mask == 1.]
    
    if masked_pred.numel() == 0 or masked_gt.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    return F.l1_loss(masked_pred, masked_gt)


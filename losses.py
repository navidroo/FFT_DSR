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
    return F.mse_loss(pred[mask == 1.], gt[mask == 1.])


def l1_loss_func(pred, gt, mask):
    """
    Compute L1 loss between prediction and ground truth, masked by the given mask.
    Handles dimension mismatches by resizing the mask to match prediction dimensions.
    """
    # Get dimensions
    _, _, pred_h, pred_w = pred.shape
    _, _, mask_h, mask_w = mask.shape
    
    # Resize mask if dimensions don't match
    if (pred_h != mask_h) or (pred_w != mask_w):
        mask = F.interpolate(mask, size=(pred_h, pred_w), mode='nearest')
    
    # Apply mask and compute loss
    return F.l1_loss(pred[mask == 1.], gt[mask == 1.])


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def CCL_pytorch(c1, warp, kernel_size=3, dilation=1):
    """
    PyTorch implementation of CCL 
    A group correlation layer that computes the correlation between two 
    feature maps.
    Args:
        c1: [batch_size, channels, height, width] - normalized feature map 1
        warp: [batch_size, channels, height, width] - normalized feature map 2
        kernel_size: int - size of the kernel for correlation
        dilation: int - dilation rate for the correlation
    Returns:
        flow: [batch_size, 2, height, width] - optical flow [flow_w, flow_h]
    """
    batch_size, channels, height, width = warp.shape
    
    pad = dilation * (kernel_size - 1) // 2
    warp_padded = F.pad(warp, [pad, pad, pad, pad])
    
    # 分成 3 x 3 的 patch 降低corrleation的计算量
    patches = F.unfold(warp_padded, kernel_size, dilation=dilation, stride=1)
    patches = patches.view(
        batch_size, channels, kernel_size*kernel_size, height, width)
    patches = patches.permute(0, 2, 1, 3, 4)  # [batch_size, 9, channels, height, width]
    
    # Vectorized correlation computation (much more efficient)
    # c1: [batch_size, channels, height, width] -> [batch_size, 1, channels, height, width]
    # patches: [batch_size, 9, channels, height, width]
    c1_expanded = c1.unsqueeze(1)  # [batch_size, 1, channels, height, width]
    match_vol = torch.sum(c1_expanded * patches, dim=2)  # [batch_size, 9, height, width]
    
    softmax_scale = 10
    match_vol = F.softmax(match_vol * softmax_scale, dim=1)
    
    flow = correlation_to_flow_pytorch(match_vol, kernel_size)
    
    return flow

def correlation_to_flow_pytorch(match_vol, kernel_size=3):
    """
    Convert correlation volume to optical flow
    Matches the TensorFlow implementation exactly
    """
    device = match_vol.device
    
    center = kernel_size // 2
    # Generate displacement vectors using meshgrid (more efficient)
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - center
    disp_h, disp_w = torch.meshgrid(coords, coords, indexing='ij')
    disp_h = disp_h.flatten()  # dy displacements
    disp_w = disp_w.flatten()  # dx displacements
    
    disp_h = disp_h.view(1, -1, 1, 1)
    disp_w = disp_w.view(1, -1, 1, 1)
    
    # Compute weighted flow
    flow_h = torch.sum(match_vol * disp_h, dim=1, keepdim=True)
    flow_w = torch.sum(match_vol * disp_w, dim=1, keepdim=True)
    
    # Concatenate [flow_w, flow_h] to match TensorFlow output order
    flow = torch.cat([flow_w, flow_h], dim=1)
    
    return flow

class CCL_Module(nn.Module):
    """
    CCL as a PyTorch module for easy integration into networks
    """
    def __init__(self):
        super(CCL_Module, self).__init__()
        
    def forward(self, feature1, feature2):
        """
        Args:
            feature1: [batch_size, channels, height, width] - feature map from image 1
            feature2: [batch_size, channels, height, width] - feature map from image 2
        Returns:
            flow: [batch_size, 2, height, width] - correlation flow
        """
        # TODO：1,缩小通道维度 2,sigmod是局部归一化，全局归一化靠softmax
        # Normalize features (as done in TensorFlow version)
        norm_feature1 = F.normalize(feature1, p=2, dim=1)
        norm_feature2 = F.normalize(feature2, p=2, dim=1)
        
        # Compute CCL
        flow = CCL_pytorch(norm_feature1, norm_feature2)
        
        return flow
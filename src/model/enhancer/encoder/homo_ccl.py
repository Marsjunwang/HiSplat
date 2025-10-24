import torch
import torch.nn as nn
from src.model.enhancer.encoder.homo_model.ccl import CCL_Module
from src.model.enhancer.encoder.homo_model.regression_net import (
    RegressionH4ptNet1, RegressionH4ptNet2, RegressionH4ptNet3,
    RegressionH4ptNetS)
from src.model.enhancer.encoder.homo_model.solve_DLT import solve_DLT
from src.model.enhancer.encoder.homo_model.normalize_homography import \
    normalize_homography
from src.model.enhancer.encoder.homo_model.transform import transform_torch

class HomoCCL(nn.Module):
    def __init__(self, 
                 ccl_mode="global", 
                 softmax_scale=10, 
                 kernel_size=3, 
                 stride=1, 
                 dilation=1, 
                 padding="same",
                 tiles_y=2, tiles_x=2,
                 patch_sizes=[16, 32, 64]):
        super(HomoCCL, self).__init__()
        self.ccl1 = CCL_Module(ccl_mode=ccl_mode, 
                              softmax_scale=softmax_scale, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              dilation=dilation,
                              padding=padding)
        self.ccl2 = CCL_Module(ccl_mode=ccl_mode, 
                              softmax_scale=softmax_scale, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              dilation=dilation*2,
                              padding=padding)
        self.ccl3 = CCL_Module(ccl_mode=ccl_mode, 
                              softmax_scale=softmax_scale, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              dilation=dilation*3,
                              padding=padding)
        self.regression_net1 = RegressionH4ptNet1(2)
        self.regression_net2 = RegressionH4ptNet2(2)
        self.regression_net3 = RegressionH4ptNet3(2)
        self.patch_sizes = patch_sizes
        # Initialize a learnable parameter for the initial homography motion 
        # (8 parameters for 4 points)
        self.init_H_motion = torch.nn.Parameter(
            torch.zeros([8, 1], device="cuda"), requires_grad=True)
    
    def forward(self, features, image_tensor):
        B = features[0].shape[0] // 2 # 现在只支持 view 2
        input_datas = [f.clone().view(B, 2, -1, f.shape[-2], f.shape[-1]) 
                       for f in features[-1:-4:-1]]
        
        H_motions = []
        H_motions.append(self.init_H_motion.repeat(B, 1, 1))
        featureflows = []
        features_warp = []
        features_warp.append(input_datas[0][:,1])
        for i, (input_data, patch_size) in enumerate(
            zip(input_datas, self.patch_sizes)):
            featureflows.append(getattr(self, f'ccl{i+1}')(
                input_data[:,0], features_warp[i]))
            H_motion = getattr(self, f'regression_net{i+1}')(featureflows[-1])
            H_motions.append(H_motion)
            
            if i == 2:
                break
            
            H = solve_DLT((H_motions[i]+H_motion)/(256//patch_size), 
                          patch_size=patch_size)
            H_norm = normalize_homography(H, patch_size)
            features_warp.append(transform_torch(input_datas[i+1][:,1], H_norm))
        
        H, W = image_tensor.shape[-2:]
        
        H1 = solve_DLT(H_motions[0]+H_motions[1], 
                       patch_size=(H, W))
        H1_norm = normalize_homography(H1, (H, W))
        H2 = solve_DLT(H_motions[0]+H_motions[1]+H_motions[2], 
                       patch_size=(H, W))
        H2_norm = normalize_homography(H2, (H, W))
        
        return H_motions
    
class HomoCCLS(nn.Module):
    """
    HomoCCL with single scale
    """
    def __init__(self, 
                 ccl_mode="global", 
                 softmax_scale=10, 
                 kernel_size=3, 
                 stride=1, 
                 dilation=1, 
                 padding="same",
                 tiles_y=2, tiles_x=2,
                 patch_sizes=[16, 32, 64],
                 **kwargs):
        super(HomoCCLS, self).__init__()
        self.ccl1 = CCL_Module(ccl_mode=ccl_mode, 
                              softmax_scale=softmax_scale, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              dilation=dilation,
                              padding=padding)
        self.regression_net1 = RegressionH4ptNetS(2)
        self.patch_sizes = patch_sizes
        # Initialize a learnable parameter for the initial homography motion 
        # (8 parameters for 4 points)
        self.init_H_motion = torch.nn.Parameter(
            torch.zeros([8, 1], device="cuda"), requires_grad=True)
    
    def forward(self, features, image_tensor):
        B = features[0].shape[0] // 2 # 现在只支持 view 2
        feat = features[-1]
        input_datas = [feat.clone().view(B, 2, -1, feat.shape[-2], 
                                         feat.shape[-1])]
        
        H_motions = []
        H_motions.append(self.init_H_motion.repeat(B, 1, 1))
        featureflows = []
        features_warp = []
        features_warp.append(input_datas[0][:,1])
        for i, (input_data, patch_size) in enumerate(
            zip(input_datas, self.patch_sizes)):
            featureflows.append(getattr(self, f'ccl{i+1}')(
                input_data[:,0], features_warp[i]))
            H_motion = getattr(self, f'regression_net{i+1}')(featureflows[-1])
            H_motions.append(H_motion)
        
        H, W = image_tensor.shape[-2:]
        H1 = solve_DLT((H_motions[i]+H_motion), patch_size=(H, W))
        M = torch.tensor([[W / 2.0, 0., W / 2.0],
                  [0., H / 2.0, H / 2.0],
                  [0., 0., 1.]], device=H1.device)
        H1_mat =torch.linalg.inv(M) @ H1 @ M
        wrap2_image = transform_torch(image_tensor[:,1], H1_mat)
        one = torch.ones_like(image_tensor[:,1])
        wrap2_one = transform_torch(one, H1_mat)
        
        return featureflows, [(wrap2_image, wrap2_one)]

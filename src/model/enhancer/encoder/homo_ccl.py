import torch
import torch.nn as nn
from src.model.enhancer.encoder.homo_model.ccl import CCL_Module
from src.model.enhancer.encoder.homo_model.regression_net import (
    RegressionH4ptNet1, RegressionH4ptNet2, RegressionH4ptNet3)
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
        self.regression_net1 = RegressionH4ptNet1()
        self.regression_net2 = RegressionH4ptNet2()
        self.regression_net3 = RegressionH4ptNet3()
        self.patch_sizes = patch_sizes
    
    def forward(self, features):
        input_datas = [f.clone() for f in features[-3::-1]]
        
        H_motions = []
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
            
            H = solve_DLT(H_motion/(256//patch_size), patch_size=patch_size)
            H_norm = normalize_homography(H, patch_size)
            features_warp.append(transform_torch(input_datas[i+1][:,1], H_norm))
        
        return H_motions

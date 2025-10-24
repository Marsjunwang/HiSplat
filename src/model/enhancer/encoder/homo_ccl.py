import torch
import torch.nn as nn
from src.model.enhancer.encoder.homo_model.ccl import CCL_Module
from src.model.enhancer.encoder.homo_model.regression_net import (
    RegressionH4ptNet1, RegressionH4ptNet2, RegressionH4ptNet3,
    RegressionH4ptNetS)
from src.model.enhancer.encoder.homo_model.solve_DLT import solve_DLT
from src.model.enhancer.encoder.homo_model.normalize_homography import \
    normalize_homography
from src.model.enhancer.encoder.homo_model.transform import (transform_torch, 
    transform_mesh_torch)
from src.model.enhancer.encoder.homo_model.h2mesh import H2Mesh

class HomoCCL(nn.Module):
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
            torch.zeros([8, 1], device="cuda"), requires_grad=False)
    
    def forward(self, features, image_tensor):
        B = features[0].shape[0] // 2 # 现在只支持 view 2
        input_datas = [f.clone().view(B, 2, -1, f.shape[-2], f.shape[-1]) 
                       for f in features[-1:-4:-1]]
        H, W = image_tensor.shape[-2:]
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
            
            H_tmp = solve_DLT((H_motions[i]+H_motion)/patch_size, 
                          patch_size=(H//patch_size, W//patch_size))

            H_norm = normalize_homography(H_tmp, (H//patch_size, W//patch_size))
            features_warp.append(transform_torch(input_datas[i+1][:,1], H_norm))
        
        H1 = solve_DLT(H_motions[0]+H_motions[1], patch_size=(H, W))
        H1_norm = normalize_homography(H1, (H, W))
        H2 = solve_DLT(H_motions[0]+H_motions[1]+H_motions[2], 
                       patch_size=(H, W))
        H2_norm = normalize_homography(H2, (H, W))
        
        wrap2_H1 = transform_torch(image_tensor[:,1], H1_norm)
        wrap2_H2 = transform_torch(image_tensor[:,1], H2_norm)
        
        one = torch.ones_like(image_tensor[:,1])
        wrap2_one_H1 = transform_torch(one, H1_norm, padding_mode='zeros') 
        wrap2_one_H2 = transform_torch(one, H2_norm, padding_mode='zeros') 
        
        ini_mesh = H2Mesh(H2, (H, W))
        mesh_motion = ini_mesh + H_motions[-1]
        
        wrap2_mesh, wrap2_one_mesh, wrap2_depth = transform_mesh_torch(image_tensor[:,1], one, one, mesh_motion)
        
        return H_motions, [(wrap2_H1, wrap2_H2), (wrap2_one_H1, wrap2_one_H2), (wrap2_mesh, wrap2_one_mesh)]
    
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
        
        # self.homo_proj = nn.Sequential(
        #         nn.Conv2d(256, 256, 1, bias=False),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU(inplace=False),
        #     )
        # self._init_weights()
    
    def _init_weights(self):
        for m in self.homo_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, feature_0, feature_1, image_tensor):
        # feat = self.homo_proj(feat)
        B = feature_0.shape[0]
        H_motions = []
        H_motions.append(self.init_H_motion.repeat(B, 1, 1))
        featureflows = []
        features_warp = []
        features_warp.append(feature_1)

        featureflows.append(getattr(self, f'ccl1')(
                feature_0, feature_1))
        H_motion = getattr(self, f'regression_net1')(featureflows[-1])
        H_motions.append(H_motion)
        
        H, W = image_tensor.shape[-2:]
        H1 = solve_DLT((H_motions[0]+H_motion), patch_size=(H, W))
        M = torch.tensor([[W / 2.0, 0., W / 2.0],
                  [0., H / 2.0, H / 2.0],
                  [0., 0., 1.]], device=H1.device)
        H1_mat = torch.linalg.inv(M) @ H1 @ M
        wrap2_image = transform_torch(image_tensor, H1_mat)
        one = torch.ones_like(image_tensor)
        wrap2_one = transform_torch(one, H1_mat, padding_mode='zeros')
        
        return featureflows, [(wrap2_image, wrap2_one)]

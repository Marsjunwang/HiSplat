import torch
import torch.nn as nn
import numpy as np

#######################################################
# Auxiliary matrices used to solve DLT
Aux_M1  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


Aux_M2  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=np.float64)



Aux_M3  = np.array([
          [0],
          [1],
          [0],
          [1],
          [0],
          [1],
          [0],
          [1]], dtype=np.float64)



Aux_M4  = np.array([
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)


Aux_M5  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)



Aux_M6  = np.array([
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ]], dtype=np.float64)


Aux_M71 = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


Aux_M72 = np.array([
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=np.float64)



Aux_M8  = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=np.float64)


Aux_Mb  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)
########################################################

class DLTSolver(nn.Module):
    """
    PyTorch DLT (Direct Linear Transform) Solver Class
    
    This class pre-initializes all auxiliary matrices on the target device for efficiency.
    Solves for homography matrix H using 4-point correspondences.
    
    Mathematical principle:
    - For each point correspondence (x,y) -> (x',y'), we get 2 linear equations
    - 4 point pairs give us 8 equations to solve for 8 unknowns (h1-h8, with h9=1)
    - Forms linear system Ax = b where x contains homography parameters
    
    Advantages of this class-based approach:
    - Pre-initialized auxiliary matrices on device (GPU/CPU)
    - No repeated tensor creation during forward pass
    - Better memory efficiency and speed
    - Easy device management
    """
    
    def __init__(self, patch_size=512., device=None):
        super(DLTSolver, self).__init__()
        
        self.patch_size = patch_size
        
        # Determine device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Pre-initialize all auxiliary matrices on device as buffers
        # Using register_buffer ensures they move with the model to different devices
        self.register_buffer('M1', torch.tensor(Aux_M1, dtype=torch.float32))
        self.register_buffer('M2', torch.tensor(Aux_M2, dtype=torch.float32))
        self.register_buffer('M3', torch.tensor(Aux_M3, dtype=torch.float32))
        self.register_buffer('M4', torch.tensor(Aux_M4, dtype=torch.float32))
        self.register_buffer('M5', torch.tensor(Aux_M5, dtype=torch.float32))
        self.register_buffer('M6', torch.tensor(Aux_M6, dtype=torch.float32))
        self.register_buffer('M71', torch.tensor(Aux_M71, dtype=torch.float32))
        self.register_buffer('M72', torch.tensor(Aux_M72, dtype=torch.float32))
        self.register_buffer('M8', torch.tensor(Aux_M8, dtype=torch.float32))
        self.register_buffer('Mb', torch.tensor(Aux_Mb, dtype=torch.float32))
        
        # Pre-initialize reference points tensor
        ref_points = torch.tensor([0., 0., patch_size, 0., 0., patch_size, patch_size, patch_size], 
                                 dtype=torch.float32).reshape(8, 1)
        self.register_buffer('ref_points', ref_points)
        
    def forward(self, pre_4pt_shift):
        """
        Solve DLT for homography matrix
        
        Args:
            pre_4pt_shift: Tensor of shape (batch_size, 8, 1) containing 4-point shifts
                          Format: [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]
                          
        Returns:
            H_mat: Homography matrices of shape (batch_size, 3, 3)
        """
        batch_size = pre_4pt_shift.shape[0]
        
        # Expand reference points for batch processing
        pts_1_tile = self.ref_points.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute target points by adding shifts to reference points
        pred_pts_2_tile = pre_4pt_shift + pts_1_tile
        
        orig_pt4 = pts_1_tile      # Original 4 points
        pred_pt4 = pred_pts_2_tile # Predicted/target 4 points
        
        # Expand auxiliary matrices for batch processing
        M1_tile = self.M1.unsqueeze(0).expand(batch_size, -1, -1)
        M2_tile = self.M2.unsqueeze(0).expand(batch_size, -1, -1)
        M3_tile = self.M3.unsqueeze(0).expand(batch_size, -1, -1)
        M4_tile = self.M4.unsqueeze(0).expand(batch_size, -1, -1)
        M5_tile = self.M5.unsqueeze(0).expand(batch_size, -1, -1)
        M6_tile = self.M6.unsqueeze(0).expand(batch_size, -1, -1)
        M71_tile = self.M71.unsqueeze(0).expand(batch_size, -1, -1)
        M72_tile = self.M72.unsqueeze(0).expand(batch_size, -1, -1)
        M8_tile = self.M8.unsqueeze(0).expand(batch_size, -1, -1)
        Mb_tile = self.Mb.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Form the linear system Ax = b to compute homography H
        # Each column of A matrix corresponds to one homography parameter
        A1 = torch.bmm(M1_tile, orig_pt4)  # Column 1: h1 coefficients
        A2 = torch.bmm(M2_tile, orig_pt4)  # Column 2: h2 coefficients
        A3 = M3_tile                       # Column 3: h3 coefficients (constant)
        A4 = torch.bmm(M4_tile, orig_pt4)  # Column 4: h4 coefficients
        A5 = torch.bmm(M5_tile, orig_pt4)  # Column 5: h5 coefficients
        A6 = M6_tile                       # Column 6: h6 coefficients (constant)
        A7 = torch.bmm(M71_tile, pred_pt4) * torch.bmm(M72_tile, orig_pt4)  # Column 7: h7 coefficients
        A8 = torch.bmm(M71_tile, pred_pt4) * torch.bmm(M8_tile, orig_pt4)   # Column 8: h8 coefficients
        
        # Stack columns to form coefficient matrix A
        # A_mat shape: (batch_size, 8, 8) - 8 equations, 8 unknowns per batch
        A_columns = [A1.reshape(-1, 8), A2.reshape(-1, 8), A3.reshape(-1, 8), A4.reshape(-1, 8),
                     A5.reshape(-1, 8), A6.reshape(-1, 8), A7.reshape(-1, 8), A8.reshape(-1, 8)]
        A_mat = torch.stack(A_columns, dim=2).transpose(1, 2)  # (batch_size, 8, 8)
        
        # Form right-hand side vector b
        b_mat = torch.bmm(Mb_tile, pred_pt4)  # (batch_size, 8, 1)
        
        # Solve the linear system Ax = b for homography parameters
        # H_8el contains the first 8 parameters of the homography matrix
        H_8el = torch.linalg.solve(A_mat, b_mat)  # (batch_size, 8, 1)
        
        # Reconstruct full 3x3 homography matrix
        # Add h9 = 1 (homogeneous coordinate normalization)
        h_ones = torch.ones(batch_size, 1, 1, dtype=torch.float32, device=H_8el.device)
        H_9el = torch.cat([H_8el, h_ones], dim=1)  # (batch_size, 9, 1)
        H_flat = H_9el.reshape(-1, 9)              # (batch_size, 9)
        H_mat = H_flat.reshape(-1, 3, 3)           # (batch_size, 3, 3)
        
        return H_mat
    
    def apply_homography(self, H_mat, points):
        """
        Apply homography transformation to points
        
        Args:
            H_mat: Homography matrices (batch_size, 3, 3)
            points: Points to transform (batch_size, N, 2)
            
        Returns:
            Transformed points (batch_size, N, 2)
        """
        batch_size, N, _ = points.shape
        
        # Convert to homogeneous coordinates
        ones = torch.ones(batch_size, N, 1, device=points.device)
        points_homo = torch.cat([points, ones], dim=2)  # (batch_size, N, 3)
        
        # Apply homography: H * p
        transformed = torch.bmm(H_mat, points_homo.transpose(1, 2))  # (batch_size, 3, N)
        transformed = transformed.transpose(1, 2)  # (batch_size, N, 3)
        
        # Convert back to Cartesian coordinates
        transformed_2d = transformed[:, :, :2] / (transformed[:, :, 2:3] + 1e-8)
        
        return transformed_2d
    
    def get_reprojection_error(self, H_mat, pts_src, pts_dst):
        """
        Calculate reprojection error for evaluation
        
        Args:
            H_mat: Homography matrices (batch_size, 3, 3)
            pts_src: Source points (batch_size, 4, 2)
            pts_dst: Destination points (batch_size, 4, 2)
            
        Returns:
            error: Mean reprojection error per batch (batch_size,)
        """
        # Apply homography to source points
        transformed_pts = self.apply_homography(H_mat, pts_src)
        
        # Calculate L2 distance
        error = torch.norm(transformed_pts - pts_dst, dim=2)  # (batch_size, 4)
        mean_error = torch.mean(error, dim=1)  # (batch_size,)
        
        return mean_error


def solve_DLT(pre_4pt_shift, patch_size=512.):
    """
    Functional interface for backward compatibility
    Creates a temporary DLTSolver instance for one-time use
    
    Note: For repeated use, it's more efficient to create a DLTSolver instance
    """
    device = pre_4pt_shift.device
    solver = DLTSolver(patch_size=patch_size, device=device)
    return solver(pre_4pt_shift)


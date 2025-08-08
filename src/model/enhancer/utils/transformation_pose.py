import torch
from jaxtyping import Float
from torch import Tensor
from einops import einsum

def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    try:
        intrinsics_inv = intrinsics.inverse()
    except:
        # For the Bug in pytorch inverse
        intrinsics_inv = intrinsics.cpu().inverse().to(intrinsics.device)

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def transformation_from_parameters(axisangle, translation, 
                                   invert=False, 
                                   intrinsics: torch.Tensor | None = None,
                                   limit_pose_to_fov_overlap: bool = False,
                                   fov_overlap_epsilon_deg: float = 0.0):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    # Optionally clamp rotation to ensure FOV overlap at infinity, controlled by config
    try:
        enable_limit = limit_pose_to_fov_overlap
    except Exception:
        enable_limit = False

    if enable_limit and intrinsics is not None:
        try:
            eps_deg = fov_overlap_epsilon_deg
        except Exception:
            eps_deg = 0.0
        R = constrain_rotation_to_infinite_fov_overlap(R, intrinsics, eps_deg)

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def _build_4x4_from_rotation(rotation_3x3: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 4x4 homogeneous matrix from a 3x3 rotation for a batch.
    rotation_3x3: [B, 3, 3]
    """
    b = rotation_3x3.shape[0]
    M = torch.zeros((b, 4, 4), device=device, dtype=dtype)
    M[:, :3, :3] = rotation_3x3
    M[:, 3, 3] = 1
    return M


def constrain_rotation_to_infinite_fov_overlap(
    rotation_matrix_4x4: torch.Tensor,
    intrinsics: torch.Tensor,
    epsilon_degrees: float = 0.0,
) -> torch.Tensor:
    """Clamp the relative rotation so that two cameras with the given intrinsics
    still have non-empty overlap of their FOVs at infinity.

    Assumes pinhole camera looking along +Z with X to the right and Y up.
    Overlap at infinity holds if |yaw| <= fov_x and |pitch| <= fov_y, where fov_x/y are full angles.

    Args:
        rotation_matrix_4x4: [B, 4, 4] relative pose rotation (homogeneous)
        intrinsics:          [B, 3, 3] camera intrinsics per batch
        epsilon_degrees:     small margin to avoid boundary, in degrees

    Returns:
        [B, 4, 4] rotation with yaw/pitch clamped to keep overlap. Roll is preserved as much as possible
        by aligning the new X axis with the projection of the original X axis onto the plane orthogonal to
        the clamped Z axis.
    """
    assert rotation_matrix_4x4.dim() == 3 and rotation_matrix_4x4.shape[-2:] == (4, 4)
    assert intrinsics.dim() == 3 and intrinsics.shape[-2:] == (3, 3)

    device = rotation_matrix_4x4.device
    dtype = rotation_matrix_4x4.dtype

    rot = rotation_matrix_4x4[:, :3, :3]

    # Forward axis of camera in the reference frame.
    ez = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(rot.shape[0], 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand(rot.shape[0], 3)

    d = torch.einsum("bij,bj->bi", rot, ez)  # [B, 3]

    # Extract horizontal and vertical deviations from optical axis.
    alpha = torch.atan2(d[:, 0], d[:, 2])  # yaw around Y, horizontal
    beta = torch.atan2(d[:, 1], d[:, 2])   # pitch around X, vertical

    fov = get_fov(intrinsics)  # [B, 2] in radians (full angles)
    fov_x = fov[:, 0]
    fov_y = fov[:, 1]

    eps = torch.deg2rad(torch.tensor(epsilon_degrees, device=device, dtype=dtype))
    limit_x = torch.clamp(fov_x - eps, min=0.0)
    limit_y = torch.clamp(fov_y - eps, min=0.0)

    alpha_clamped = torch.clamp(alpha, min=-limit_x, max=limit_x)
    beta_clamped = torch.clamp(beta, min=-limit_y, max=limit_y)

    # Reconstruct a clamped forward direction using the clamped angles.
    # Using relations: tan(alpha) = x/z, tan(beta) = y/z. Use z=1 for direction then normalize.
    x_c = torch.tan(alpha_clamped)
    y_c = torch.tan(beta_clamped)
    ones = torch.ones_like(x_c)
    z_cand = torch.stack([x_c, y_c, ones], dim=-1)
    z_c = z_cand / (z_cand.norm(dim=-1, keepdim=True) + 1e-9)

    # Preserve roll as much as possible by projecting the original X axis onto the plane normal to z_c.
    x_old = torch.einsum("bij,bj->bi", rot, ex)
    x_proj = x_old - (x_old * z_c).sum(dim=-1, keepdim=True) * z_c
    x_proj_norm = x_proj.norm(dim=-1, keepdim=True)

    # Fallback basis if projection is degenerate (close to zero length)
    y_temp = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand(rot.shape[0], 3)
    parallel = (torch.abs((y_temp * z_c).sum(dim=-1)) > 0.99)
    y_temp = torch.where(parallel.unsqueeze(-1), torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(y_temp), y_temp)

    use_fallback = (x_proj_norm.squeeze(-1) < 1e-6)
    # Compute orthonormal basis
    x_c_vec = torch.where(
        use_fallback.unsqueeze(-1),
        torch.nn.functional.normalize(torch.cross(y_temp, z_c, dim=-1), dim=-1),
        torch.nn.functional.normalize(x_proj, dim=-1),
    )
    y_c_vec = torch.nn.functional.normalize(torch.cross(z_c, x_c_vec, dim=-1), dim=-1)

    rot_c = torch.stack([x_c_vec, y_c_vec, z_c], dim=-1)  # [B, 3, 3] columns are axes
    return _build_4x4_from_rotation(rot_c, device, dtype)



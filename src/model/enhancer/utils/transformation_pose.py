import torch

def _ensure_intrinsics_3x3(K: torch.Tensor) -> torch.Tensor:
    """Return a 3x3 intrinsics tensor by slicing if a 4x4 is provided."""
    if K.shape[-2:] == (4, 4):
        return K[..., :3, :3]
    return K


def get_fov(intrinsics: torch.Tensor, image_size: tuple[int, int] | None = None) -> torch.Tensor:
    """Compute full FOV angles (x, y) in radians for intrinsics with shape [..., 3, 3].
    Supports both normalized intrinsics (cx≈0.5, cy≈0.5) and pixel intrinsics when image_size is given.
    """
    intrinsics = _ensure_intrinsics_3x3(intrinsics)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    use_normalized = torch.isfinite(cx).all() and torch.isfinite(cy).all() and (
        (cx - 0.5).abs().mean() < 0.05 and (cy - 0.5).abs().mean() < 0.05
    )

    if image_size is not None and not use_normalized:
        # Pixel intrinsics case
        W, H = image_size[1], image_size[0]
        fov_x = 2 * torch.atan(0.5 * W / (fx + 1e-9))
        fov_y = 2 * torch.atan(0.5 * H / (fy + 1e-9))
        return torch.stack((fov_x, fov_y), dim=-1)

    # Fallback: treat as normalized intrinsics and derive via K^{-1}
    try:
        intrinsics_inv = intrinsics.inverse()
    except Exception:
        intrinsics_inv = intrinsics.cpu().inverse().to(intrinsics.device)

    def process_vector(vec3):
        v = torch.tensor(vec3, dtype=torch.float32, device=intrinsics.device)
        v = torch.einsum("...ij,j->...i", intrinsics_inv, v)
        return v / (v.norm(dim=-1, keepdim=True) + 1e-9)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)

def get_translation_matrix(translation_vector: torch.Tensor) -> torch.Tensor:
    """Convert a translation vector into a 4x4 transformation matrix.
    Accepts shapes [..., 3] or [..., 1, 3]. Returns [..., 4, 4].
    """
    if translation_vector.dim() >= 2 and translation_vector.shape[-2:] == (1, 3):
        translation_vector = translation_vector.squeeze(-2)
    elif translation_vector.dim() >= 2 and translation_vector.shape[-2:] == (3, 1):
        translation_vector = translation_vector.squeeze(-1)
    assert translation_vector.shape[-1] == 3, "translation_vector must have last dim = 3"

    batch_shape = translation_vector.shape[:-1]
    device = translation_vector.device
    dtype = translation_vector.dtype

    T = torch.zeros((*batch_shape, 4, 4), device=device, dtype=dtype)
    T[..., 0, 0] = 1
    T[..., 1, 1] = 1
    T[..., 2, 2] = 1
    T[..., 3, 3] = 1

    t = translation_vector.contiguous().view(*batch_shape, 3, 1)
    T[..., :3, 3:4] = t
    return T

def rot_from_axisangle(vec: torch.Tensor) -> torch.Tensor:
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    # Make function robust to inputs shaped [..., 3] or [..., 1, 3]
    if vec.shape[-2:] == (1, 3) and vec.shape[-3] == 1:
        vec = vec.squeeze(-3).squeeze(-2)
    if vec.shape[-1] != 3:
        raise ValueError(f"rot_from_axisangle expects last dimension 3, got {tuple(vec.shape)}")

    angle = torch.norm(vec, dim=-1, keepdim=True)  # [..., 1]
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0:1]
    y = axis[..., 1:1+1]
    z = axis[..., 2:2+1]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    batch_shape = vec.shape[:-1]
    rot = torch.zeros((*batch_shape, 4, 4), device=vec.device, dtype=vec.dtype)

    # Squeeze only trailing singleton dims to preserve batch dims
    rot[..., 0, 0] = (x * xC + ca).squeeze(-1)
    rot[..., 0, 1] = (xyC - zs).squeeze(-1)
    rot[..., 0, 2] = (zxC + ys).squeeze(-1)
    rot[..., 1, 0] = (xyC + zs).squeeze(-1)
    rot[..., 1, 1] = (y * yC + ca).squeeze(-1)
    rot[..., 1, 2] = (yzC - xs).squeeze(-1)
    rot[..., 2, 0] = (zxC - ys).squeeze(-1)
    rot[..., 2, 1] = (yzC + xs).squeeze(-1)
    rot[..., 2, 2] = (z * zC + ca).squeeze(-1)
    rot[..., 3, 3] = 1

    return rot

def transformation_from_parameters(
    axisangle: torch.Tensor,
    translation: torch.Tensor,
    invert: bool = False,
    intrinsics: torch.Tensor | None = None,
    near: torch.Tensor | None = None,
    far: torch.Tensor | None = None,
    image_size: tuple[int, int] | None = None,
    limit_pose_to_fov_overlap: bool = False,
    fov_overlap_epsilon_deg: float = 0.5,
    fov_overlap_mode: str = "infinite",  # "infinite" | "finite"
    fov_depth_mode: str = "any",  # "any" | "all" | "near_only" | "far_only"
) -> torch.Tensor:
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """  
    R = rot_from_axisangle(axisangle)[..., :3, :3]
    t = translation.clone()
    # Normalize translation shape to [..., 3]
    if t.shape[-2:] == (1, 3):
        t = t.squeeze(-2)
    elif t.shape[-2:] == (3, 1):
        t = t.squeeze(-1)

    if invert:
        R = R.transpose(-2, -1)
        # Avoid in-place to keep autograd graph valid
        t = -t

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
        # Compose current relative pose [R|t]
        M_curr = torch.zeros((*R.shape[:-2], 4, 4), device=R.device, dtype=R.dtype)
        M_curr[..., :3, :3] = R[..., :3, :3]
        M_curr[..., :3, 3] = t[..., :3]
        M_curr[..., 3, 3] = 1
        # Constrain pose (rotation + translation)
        M_curr = constrain_pose_to_fov_overlap(
            M_curr,
            intrinsics,
            near=near,
            far=far,
            image_size=image_size,
            epsilon_degrees=eps_deg,
            mode=fov_overlap_mode,
            depth_mode=fov_depth_mode,
        )
        # Decompose back
        R = M_curr[..., :3, :3]
        t = M_curr[..., :3, 3:4]

    T = get_translation_matrix(t)
    R_h = _build_4x4_from_rotation(R, device=R.device, dtype=R.dtype)

    if invert:
        M = torch.matmul(R_h, T)
    else:
        M = torch.matmul(T, R_h)

    return M


def _build_4x4_from_rotation(rotation_3x3: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 4x4 homogeneous matrix from a 3x3 rotation for arbitrary batch dims.
    rotation_3x3: [..., 3, 3]
    """
    batch_shape = rotation_3x3.shape[:-2]
    M = torch.zeros((*batch_shape, 4, 4), device=device, dtype=dtype)
    M[..., :3, :3] = rotation_3x3
    M[..., 3, 3] = 1
    return M


def constrain_rotation_to_infinite_fov_overlap(
    rotation_matrix_4x4: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor | None = 0.01,
    far: torch.Tensor | None = 1000.,
    image_size: tuple[int, int] | None = (256, 256),
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
    assert rotation_matrix_4x4.shape[-2:] == (4, 4)
    intrinsics = _ensure_intrinsics_3x3(intrinsics)
    assert intrinsics.shape[-2:] == (3, 3)

    device = rotation_matrix_4x4.device
    dtype = rotation_matrix_4x4.dtype

    batch_shape = rotation_matrix_4x4.shape[:-2]

    # Broadcast intrinsics to match batch_shape if needed
    if intrinsics.shape[:-2] != batch_shape:
        pad = len(batch_shape) - len(intrinsics.shape[:-2])
        intrinsics = intrinsics.reshape((1,) * pad + intrinsics.shape).expand(*batch_shape, 3, 3)

    rot = rotation_matrix_4x4[..., :3, :3]

    # Forward axis of camera in the reference frame.
    ez = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(*batch_shape, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand(*batch_shape, 3)

    d = torch.einsum("...ij,...j->...i", rot, ez)  # [..., 3]

    # Extract horizontal and vertical deviations from optical axis.
    alpha = torch.atan2(d[..., 0], d[..., 2])  # yaw around Y, horizontal
    beta = torch.atan2(d[..., 1], d[..., 2])   # pitch around X, vertical

    fov = get_fov(intrinsics, image_size=image_size)  # [..., (view), 2] in radians (full angles)
    # If intrinsics provided per-view (e.g., shape [..., V, 3, 3]),
    # use the intersection of FOVs by taking the per-axis minimum across views.
    if fov.dim() >= 2 and fov.shape[-2] > 1:
        fov = fov.min(dim=-2).values  # [..., 2]
    fov_x = fov[..., 0]
    fov_y = fov[..., 1]

    eps = torch.deg2rad(torch.tensor(epsilon_degrees, device=device, dtype=dtype))
    # Use smooth squashing instead of hard clamp for differentiability
    limit_x = torch.clamp(fov_x - eps, min=1e-6)
    limit_y = torch.clamp(fov_y - eps, min=1e-6)
    alpha_s = limit_x * torch.tanh(alpha / (limit_x + 1e-9))
    beta_s = limit_y * torch.tanh(beta / (limit_y + 1e-9))

    # Reconstruct a squashed forward direction
    x_c = torch.tan(alpha_s)
    y_c = torch.tan(beta_s)
    ones = torch.ones_like(x_c)
    z_cand = torch.stack([x_c, y_c, ones], dim=-1)
    z_c = z_cand / (z_cand.norm(dim=-1, keepdim=True) + 1e-9)

    # Preserve roll as much as possible by projecting the original X axis onto the plane normal to z_c.
    x_old = torch.einsum("...ij,...j->...i", rot, ex)
    x_proj = x_old - (x_old * z_c).sum(dim=-1, keepdim=True) * z_c
    x_proj_norm = x_proj.norm(dim=-1, keepdim=True)

    # Fallback basis if projection is degenerate (close to zero length)
    y_temp = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand(*batch_shape, 3)
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

    rot_c = torch.stack([x_c_vec, y_c_vec, z_c], dim=-1)  # [..., 3, 3] columns are axes
    return _build_4x4_from_rotation(rot_c, device, dtype)


def constrain_rotation_to_finite_fov_overlap(
    rotation_matrix_4x4: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor | None = 0.01,
    far: torch.Tensor | None = 1000.,
    image_size: tuple[int, int] | None = (256, 256),
    epsilon_degrees: float = 0.0,
) -> torch.Tensor:
    """Differentiable finite-depth-friendly rotation constraint.

    Uses the same smooth yaw/pitch squashing as the infinite-depth variant for
    efficiency and differentiability. Finite-depth effects are handled at the
    full-pose level.
    """
    return constrain_rotation_to_infinite_fov_overlap(
        rotation_matrix_4x4,
        intrinsics,
        near=near,
        far=far,
        image_size=image_size,
        epsilon_degrees=epsilon_degrees,
    )


def constrain_rotation_to_fov_overlap(
    rotation_matrix_4x4: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor | None = 0.01,
    far: torch.Tensor | None = 1000.,
    image_size: tuple[int, int] | None = (256, 256),
    epsilon_degrees: float = 0.0,
    mode: str = "infinite",
) -> torch.Tensor:
    """Wrapper to choose between infinite-depth and finite-depth FOV-overlap constraints.

    Args:
        mode: "infinite" or "finite".
    """
    if mode == "finite":
        return constrain_rotation_to_finite_fov_overlap(
            rotation_matrix_4x4,
            intrinsics,
            near=near,
            far=far,
            image_size=image_size,
            epsilon_degrees=epsilon_degrees,
        )
    # Default to infinite-depth
    return constrain_rotation_to_infinite_fov_overlap(
        rotation_matrix_4x4,
        intrinsics,
        near=near,
        far=far,
        image_size=image_size,
        epsilon_degrees=epsilon_degrees,
    )


def constrain_pose_to_fov_overlap(
    transform_matrix_4x4: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor | None = 0.01,
    far: torch.Tensor | None = 1000.,
    image_size: tuple[int, int] | None = (256, 256),
    epsilon_degrees: float = 0.0,
    mode: str = "infinite",
    depth_mode: str = "any",
) -> torch.Tensor:
    """Differentiable full-pose constraint with only two modes: "infinite" and "finite".

    - infinite: smooth rotation squashing; translation unchanged.
    - finite:   smooth joint rotation+translation adjustment via soft overlap.
    """
    assert transform_matrix_4x4.shape[-2:] == (4, 4)
    intrinsics = _ensure_intrinsics_3x3(intrinsics)

    if mode == "finite":
        return constrain_pose_to_fov_overlap_smooth(
            transform_matrix_4x4,
            intrinsics,
            near=near,
            far=far,
            image_size=image_size,
            epsilon_degrees=epsilon_degrees,
            depth_mode=depth_mode,
        )

    # Infinite-depth: only adjust rotation smoothly, keep translation
    device = transform_matrix_4x4.device
    dtype = transform_matrix_4x4.dtype
    R = transform_matrix_4x4[..., :3, :3]
    t = transform_matrix_4x4[..., :3, 3]

    R_h = _build_4x4_from_rotation(R, device, dtype)
    R_h = constrain_rotation_to_fov_overlap(
        R_h,
        intrinsics,
        near=near,
        far=far,
        image_size=image_size,
        epsilon_degrees=epsilon_degrees,
        mode="infinite",
    )
    R_new = R_h[..., :3, :3]

    M = torch.zeros((*transform_matrix_4x4.shape[:-2], 4, 4), device=device, dtype=dtype)
    M[..., :3, :3] = R_new
    M[..., :3, 3] = t
    M[..., 3, 3] = 1
    return M
## end of helpers


def constrain_pose_to_fov_overlap_smooth(
    transform_matrix_4x4: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor | None = 0.01,
    far: torch.Tensor | None = 1000.,
    image_size: tuple[int, int] | None = (256, 256),
    epsilon_degrees: float = 0.0,
    depth_mode: str = "any",
) -> torch.Tensor:
    """Differentiable joint constraint of rotation and translation to keep finite-distance FOV overlap.

    - Uses smooth angle squashing and soft in-bounds scores (sigmoid) to avoid vanishing gradients.
    - Blends original pose with clamped pose based on continuous violation weights, not hard clipping.

    Assumptions: pinhole camera looks along +Z, X right, Y up.
    """
    assert transform_matrix_4x4.shape[-2:] == (4, 4)
    intrinsics = _ensure_intrinsics_3x3(intrinsics)
    assert intrinsics.shape[-2:] == (3, 3)

    device = transform_matrix_4x4.device
    dtype = transform_matrix_4x4.dtype

    batch_shape = transform_matrix_4x4.shape[:-2]

    # Broadcast intrinsics to match batch_shape if needed
    if intrinsics.shape[:-2] != batch_shape:
        pad = len(batch_shape) - len(intrinsics.shape[:-2])
        intrinsics = intrinsics.reshape((1,) * pad + intrinsics.shape).expand(*batch_shape, 3, 3)

    # Split R, t
    R = transform_matrix_4x4[..., :3, :3]
    t = transform_matrix_4x4[..., :3, 3]

    # FOV limits
    fov = get_fov(intrinsics, image_size=image_size)  # [..., (view), 2]
    if fov.dim() >= 2 and fov.shape[-2] > 1:
        fov = fov.min(dim=-2).values
    fov_x = fov[..., 0]
    fov_y = fov[..., 1]
    eps = torch.deg2rad(torch.tensor(epsilon_degrees, device=device, dtype=dtype))
    # Keep small positive to avoid divide-by-zero in angle scaling
    limit_x = torch.clamp(fov_x - eps, min=1e-4)
    limit_y = torch.clamp(fov_y - eps, min=1e-4)

    # Camera axes
    ez = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(*batch_shape, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand(*batch_shape, 3)

    # Current forward vector and yaw/pitch (alpha horizontal, beta vertical)
    d = torch.einsum("...ij,...j->...i", R, ez)
    alpha = torch.atan2(d[..., 0], d[..., 2])
    beta = torch.atan2(d[..., 1], d[..., 2])

    # Smooth angle squash within limits using tanh, then blend by soft violation
    angle_soft_k = torch.tensor(6.0, device=device, dtype=dtype)  # softness for angle violation
    alpha_squash = limit_x * torch.tanh(alpha / (limit_x + 1e-6))
    beta_squash = limit_y * torch.tanh(beta / (limit_y + 1e-6))
    # Angle violation (soft, always >=0)
    alpha_excess = torch.nn.functional.relu(torch.abs(alpha) - limit_x)
    beta_excess = torch.nn.functional.relu(torch.abs(beta) - limit_y)
    # Combine violations smoothly (sum is simple and smooth)
    rot_violation = alpha_excess + beta_excess
    w_rot = torch.sigmoid(angle_soft_k * rot_violation)  # in (0,1)

    # Build squashed rotation from (alpha_s, beta_s) preserving roll via projecting old X
    def build_R_from_angles(alpha_like: torch.Tensor, beta_like: torch.Tensor, R_ref: torch.Tensor) -> torch.Tensor:
        x_c = torch.tan(alpha_like)
        y_c = torch.tan(beta_like)
        ones = torch.ones_like(x_c)
        z_cand = torch.stack([x_c, y_c, ones], dim=-1)
        z_c = z_cand / (z_cand.norm(dim=-1, keepdim=True) + 1e-9)
        x_old = torch.einsum("...ij,...j->...i", R_ref, ex)
        x_proj = x_old - (x_old * z_c).sum(dim=-1, keepdim=True) * z_c
        x_proj_norm = x_proj.norm(dim=-1, keepdim=True)
        y_temp = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand_as(x_proj)
        parallel = (torch.abs((y_temp * z_c).sum(dim=-1)) > 0.99)
        y_temp = torch.where(
            parallel.unsqueeze(-1),
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(y_temp),
            y_temp,
        )
        use_fallback = (x_proj_norm.squeeze(-1) < 1e-6)
        x_c_vec = torch.where(
            use_fallback.unsqueeze(-1),
            torch.nn.functional.normalize(torch.cross(y_temp, z_c, dim=-1), dim=-1),
            torch.nn.functional.normalize(x_proj, dim=-1),
        )
        y_c_vec = torch.nn.functional.normalize(torch.cross(z_c, x_c_vec, dim=-1), dim=-1)
        return torch.stack([x_c_vec, y_c_vec, z_c], dim=-1)

    R_squash = build_R_from_angles(alpha_squash, beta_squash, R)

    # Soft in-bounds score for points to drive translation shrinkage (and joint weight)
    def is_normalized_K(K: torch.Tensor) -> torch.Tensor:
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        return ((cx - 0.5).abs() < 0.05) & ((cy - 0.5).abs() < 0.05)

    if intrinsics.dim() >= 4 and intrinsics.shape[-4] > 1:
        K_a = intrinsics[..., 0, :, :]
        K_b = intrinsics[..., 1, :, :]
    else:
        K_a = intrinsics
        K_b = intrinsics

    norm_a = is_normalized_K(K_a)
    norm_b = is_normalized_K(K_b)
    if image_size is None:
        H, W = 1.0, 1.0
    else:
        H, W = float(image_size[0]), float(image_size[1])

    pix_soft_k = torch.tensor(6.0, device=device, dtype=dtype)
    margin = torch.tensor(0.0, device=device, dtype=dtype)  # extra inside margin if needed

    def soft_inbounds_score(K: torch.Tensor, p_cam: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
        # Broadcast intrinsics over the sample dimension of p_cam (..., S)
        fx = K[..., 0, 0][..., None]
        fy = K[..., 1, 1][..., None]
        cx = K[..., 0, 2][..., None]
        cy = K[..., 1, 2][..., None]
        z = p_cam[..., 2]
        u = fx * (p_cam[..., 0] / (z + 1e-9)) + cx
        v = fy * (p_cam[..., 1] / (z + 1e-9)) + cy
        if image_size is None:
            umin = torch.tensor(0.0, device=device, dtype=dtype)[None]
            umax = torch.tensor(1.0, device=device, dtype=dtype)[None]
            vmin = torch.tensor(0.0, device=device, dtype=dtype)[None]
            vmax = torch.tensor(1.0, device=device, dtype=dtype)[None]
        else:
            umin = torch.where(normalized, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype))[..., None]
            vmin = torch.where(normalized, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype))[..., None]
            umax = torch.where(normalized, torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(W, device=device, dtype=dtype))[..., None]
            vmax = torch.where(normalized, torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(H, device=device, dtype=dtype))[..., None]
        su = torch.sigmoid(pix_soft_k * (u - umin + margin)) * torch.sigmoid(pix_soft_k * (umax - u + margin))
        sv = torch.sigmoid(pix_soft_k * (v - vmin + margin)) * torch.sigmoid(pix_soft_k * (vmax - v + margin))
        sz = torch.sigmoid(pix_soft_k * p_cam[..., 2])
        return su * sv * sz

    # Sample rays across the image to detect overlap even when centers are outside
    def build_sample_rays(K: torch.Tensor, normalized: torch.Tensor, N: int = 3) -> torch.Tensor:
        # returns [..., M, 3] unit rays (M=N*N) in camera frame
        if image_size is None or normalized.all():
            grid = torch.linspace(0.1, 0.9, N, device=device, dtype=dtype)
            uu, vv = torch.meshgrid(grid, grid, indexing="xy")
        else:
            grid_u = torch.linspace(0.1 * W, 0.9 * W, N, device=device, dtype=dtype)
            grid_v = torch.linspace(0.1 * H, 0.9 * H, N, device=device, dtype=dtype)
            uu, vv = torch.meshgrid(grid_u, grid_v, indexing="xy")
        ones = torch.ones_like(uu)
        pix = torch.stack([uu, vv, ones], dim=-1).reshape(1, N * N, 3)
        pix = pix.expand(K.shape[:-2] + (N * N, 3))
        Kinv = K.inverse()
        dirs = torch.einsum("...ij,...nj->...ni", Kinv, pix)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
        return dirs

    def get_depth_samples() -> torch.Tensor:
        # returns [..., K] depending on depth_mode
        if isinstance(near, torch.Tensor):
            z0 = near.reshape(batch_shape)
        else:
            z0 = torch.full(batch_shape, float(near if near is not None else 0.01), device=device, dtype=dtype)
        if isinstance(far, torch.Tensor):
            z2 = far.reshape(batch_shape)
        else:
            z2 = torch.full(batch_shape, float(far if far is not None else 1000.0), device=device, dtype=dtype)
        if depth_mode == "near_only":
            return z0.unsqueeze(-1)
        if depth_mode == "far_only":
            return z2.unsqueeze(-1)
        if depth_mode == "all":
            # denser sampling near and far
            z_quart = 0.25 * z0 + 0.75 * z2
            z_mid = 0.5 * (z0 + z2)
            z_34 = 0.75 * z0 + 0.25 * z2
            return torch.stack([z0, z_34, z_mid, z_quart, z2], dim=-1)
        # default: any -> near, mid, far
        z1 = 0.5 * (z0 + z2)
        return torch.stack([z0, z1, z2], dim=-1)

    def soft_overlap_score_A_to_B(R_ab: torch.Tensor, t_ab: torch.Tensor) -> torch.Tensor:
        rays_a = build_sample_rays(K_a, norm_a, N=3)  # [..., M, 3]
        z = get_depth_samples()  # [..., 3]
        # Points in A at multiple depths per ray: [..., M, 3depths, 3]
        # rays_a: [..., M, 3], z: [..., K] → expand z to [..., 1, K, 1] so result is [..., M, K, 3]
        z_expanded = z.unsqueeze(-2).unsqueeze(-1)
        P_a = rays_a.unsqueeze(-2) * z_expanded
        # flatten depth and samples: [..., M*K, 3]
        P_a = P_a.reshape(batch_shape + (P_a.shape[-3] * P_a.shape[-2], 3))
        # Transform to B: X_b = R_ab X_a + t_ab
        P_b = torch.einsum("...ij,...nj->...ni", R_ab, P_a) + t_ab.unsqueeze(-2)
        # Soft in-bounds per point
        s = soft_inbounds_score(K_b, P_b, norm_b)  # [..., M*3]
        s_clamped = torch.clamp(s, 1e-6, 1.0 - 1e-6)
        if depth_mode == "all":
            # require consistent overlap across all sampled depths -> Soft-AND
            soft_and = torch.exp(torch.sum(torch.log(s_clamped), dim=-1))
            return soft_and
        # default/any: Soft-OR aggregation: 1 - prod(1 - s)
        soft_any = 1.0 - torch.exp(torch.sum(torch.log(1.0 - s_clamped), dim=-1))
        return soft_any

    # A->B uses (R, t). B->A uses inverse (R^T, -R^T t)
    score_A_to_B = soft_overlap_score_A_to_B(R, t)
    R_T = R.transpose(-2, -1)
    t_inv = -torch.einsum("...ij,...j->...i", R_T, t)
    # swap intrinsics/norm flags for B as source and A as dest
    K_a, K_b, norm_a, norm_b = K_b, K_a, norm_b, norm_a
    score_B_to_A = soft_overlap_score_A_to_B(R_T, t_inv)
    # restore original K_a/K_b variables are no longer needed after this point

    # Combine both directions
    overlap_score = 0.5 * (score_A_to_B + score_B_to_A)

    # Joint weight: if rotation violates or overlap is low, increase weight
    # Probabilistic sum keeps in (0,1) smoothly
    w_joint = 1.0 - (1.0 - w_rot) * overlap_score

    # Blend angles and rebuild rotation
    alpha_new = (1.0 - w_joint) * alpha + w_joint * alpha_squash
    beta_new = (1.0 - w_joint) * beta + w_joint * beta_squash
    R_new = build_R_from_angles(alpha_new, beta_new, R)

    # Smoothly shrink translation as overlap drops
    # Map high overlap -> keep t ; low overlap -> shrink toward 0
    t_new = (overlap_score.unsqueeze(-1)) * t

    # Pack back matrix
    M = torch.zeros((*batch_shape, 4, 4), device=device, dtype=dtype)
    M[..., :3, :3] = R_new
    M[..., :3, 3] = t_new
    M[..., 3, 3] = 1
    return M
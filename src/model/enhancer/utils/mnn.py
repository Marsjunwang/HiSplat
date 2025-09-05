import torch

def hard_mnn(f0: torch.Tensor=None, 
             f1: torch.Tensor=None, 
             scores: torch.Tensor=None
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    硬相互最近邻匹配
    Args:
        f0: 特征张量，形状 [N0, D] 或 [B, N0, D]
        f1: 特征张量，形状 [N1, D] 或 [B, N1, D]  
        scores: 预计算的相似度矩阵，形状 [N0, N1] 或 [B, N0, N1]
    Returns:
        M: 匹配矩阵
        match12: f0到f1的匹配索引
        match21: f1到f0的匹配索引
    """
    if f0 is not None and f1 is not None:
        if f0.dim() == 3:  # batch mode
            S = torch.bmm(f0, f1.transpose(-2, -1))
        else:  # single mode
            S = f0 @ f1.t()
    elif scores is not None:
        S = scores
    else:
        raise ValueError("Either f0 and f1 or scores must be provided")
    
    if S.dim() == 3:  # batch mode
        return _hard_mnn_batch(S)
    else:  # single mode
        return _hard_mnn_single(S)

def _hard_mnn_single(S: torch.Tensor):
    """单个样本的硬相互最近邻"""
    match12 = S.argmax(dim=-1)
    match21 = S.argmax(dim=-2)
    N0 = S.shape[-1]
    idx0 = torch.arange(N0, device=S.device)
    mutual = match21[match12] == idx0
    M = torch.zeros_like(S)
    M[idx0[mutual], match12[mutual]] = 1.0
    return M, match12, match21

def _hard_mnn_batch(S: torch.Tensor):
    """批量样本的硬相互最近邻"""
    B, N0, N1 = S.shape
    match12 = S.argmax(dim=-1)  # [B, N0] - f0中每个点在f1中的最佳匹配
    match21 = S.argmax(dim=-2)  # [B, N1] - f1中每个点在f0中的最佳匹配
    
    M = torch.zeros_like(S)  # [B, N0, N1]
    
    batch_idx = torch.arange(B, device=S.device).unsqueeze(1)  # [B, 1]
    point_idx = torch.arange(N0, device=S.device).unsqueeze(0).expand(B, -1)
    
    valid_matches = match12 < N1  # [B, N0] - 确保match12索引有效
    
    reverse_matches = torch.zeros_like(match12)  # [B, N0]
    reverse_matches[valid_matches] = match21[batch_idx.expand(-1, N0)[valid_matches], 
                                            match12[valid_matches]]
    
    mutual = valid_matches & (reverse_matches == point_idx)  # [B, N0]
    
    M[batch_idx.expand(-1, N0)[mutual], 
      point_idx[mutual], 
      match12[mutual]] = 1.0
    match12[~mutual] = -1
    match21[~mutual] = -1
    
    return M, match12, match21

def soft_mnn(f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
    """
    软相互最近邻匹配
    Args:
        f0: 特征张量，形状 [N0, D] 或 [B, N0, D]
        f1: 特征张量，形状 [N1, D] 或 [B, N1, D]
    Returns:
        匹配概率矩阵，形状 [N0, N1] 或 [B, N0, N1]
    """
    if f0.dim() == 3:  # batch mode
        S = torch.bmm(f0, f1.transpose(-2, -1))
        P12 = torch.softmax(S, dim=-1)
        P21 = torch.softmax(S.transpose(-2, -1), dim=-1)
        return P12 * P21.transpose(-2, -1)
    else:  # single mode
        S = f0 @ f1.t()
        P12 = torch.softmax(S, dim=-1)
        P21 = torch.softmax(S.t(), dim=-1)
        return P12 * P21.t()

def soft_mnn_with_tau(f0: torch.Tensor, 
                      f1: torch.Tensor, 
                      tau: float = 0.2) -> torch.Tensor:
    """
    带温度系数的软相互最近邻匹配
    Args:
        f0: 特征张量，形状 [N0, D] 或 [B, N0, D]
        f1: 特征张量，形状 [N1, D] 或 [B, N1, D]
        tau: 温度系数，越小越"硬"
    Returns:
        匹配概率矩阵，形状 [N0, N1] 或 [B, N0, N1]
    """
    if f0.dim() == 3:  # batch mode
        S = torch.bmm(f0, f1.transpose(-2, -1))
        # 数值稳定的softmax：减去最大值
        S_stable = S - S.max(dim=-1, keepdim=True).values
        P12 = torch.softmax(S_stable / tau, dim=-1)
        S_stable_t = S.transpose(-2, -1) - S.transpose(-2, -1).max(dim=-1, keepdim=True).values
        P21 = torch.softmax(S_stable_t / tau, dim=-1)
        return P12 * P21.transpose(-2, -1)
    else:  # single mode
        S = f0 @ f1.t()
        # 数值稳定的softmax：减去最大值
        S_stable = S - S.max(dim=-1, keepdim=True).values
        P12 = torch.softmax(S_stable / tau, dim=-1)
        S_stable_t = S.t() - S.t().max(dim=-1, keepdim=True).values
        P21 = torch.softmax(S_stable_t / tau, dim=-1)
        return P12 * P21.t()

def topk_soft_mnn_with_tau(f0: torch.Tensor, 
                           f1: torch.Tensor, 
                           tau: float = 0.2, 
                           k: int = 50) -> torch.Tensor:
    """
    Top-K软相互最近邻匹配（稀疏版本）
    Args:
        f0: 特征张量，形状 [N0, D] 或 [B, N0, D]
        f1: 特征张量，形状 [N1, D] 或 [B, N1, D]
        tau: 温度系数
        k: 每行/列保留的top-k个值
    Returns:
        匹配概率矩阵，形状 [N0, N1] 或 [B, N0, N1]
    """
    if f0.dim() == 3:  # batch mode
        S = torch.bmm(f0, f1.transpose(-2, -1))
        B, N0, N1 = S.shape
        
        # 行方向top-k
        v, i = S.topk(k, dim=-1)  # [B, N0, k]
        row = torch.full_like(S, float('-inf'))
        row.scatter_(-1, i, v)
        P12 = torch.softmax(row / tau, dim=-1)
        
        # 列方向top-k
        S_t = S.transpose(-2, -1)  # [B, N1, N0]
        v2, i2 = S_t.topk(k, dim=-1)  # [B, N1, k]
        col = torch.full_like(S_t, float('-inf'))
        col.scatter_(-1, i2, v2)
        P21 = torch.softmax(col / tau, dim=-1)
        
        return P12 * P21.transpose(-2, -1)
    else:  # single mode
        S = f0 @ f1.t()
        # 行方向
        v, i = S.topk(k, dim=-1)
        row = torch.full_like(S, float('-inf'))
        row.scatter_(1, i, v)
        P12 = torch.softmax(row / tau, dim=-1)
        # 列方向
        v2, i2 = S.t().topk(k, dim=-1)
        col = torch.full_like(S.t(), float('-inf'))
        col.scatter_(1, i2, v2)
        P21 = torch.softmax(col / tau, dim=-1)
        return P12 * P21.t() 


# 使用示例：
# 
# 对于您的输入形状 [2, 1024, 1024]（batch=2, 两组特征都是1024维的1024个点）:
# 
# # 方法1: 直接使用特征向量
# f0 = torch.randn(2, 1024, 1024)  # batch=2, N0=1024个点, D=1024维特征
# f1 = torch.randn(2, 1024, 1024)  # batch=2, N1=1024个点, D=1024维特征
# M, match12, match21 = hard_mnn(f0, f1)
# # M: [2, 1024, 1024] 匹配矩阵
# # match12: [2, 1024] f0中每个点在f1中的最佳匹配索引
# # match21: [2, 1024] f1中每个点在f0中的最佳匹配索引
#
# # 方法2: 使用预计算的相似度矩阵
# scores = torch.randn(2, 1024, 1024)  # 预计算的相似度矩阵
# M, match12, match21 = hard_mnn(scores=scores)
#
# # 软匹配版本
# soft_matches = soft_mnn_with_tau(f0, f1, tau=0.1)  # [2, 1024, 1024]
# 
# # 稀疏版本（只考虑top-50匹配）
# sparse_matches = topk_soft_mnn_with_tau(f0, f1, tau=0.1, k=50)  # [2, 1024, 1024]

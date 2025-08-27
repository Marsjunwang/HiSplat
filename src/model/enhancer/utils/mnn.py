import torch

def hard_mnn(f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
    S = f0 @ f1.t()
    match12 = S.argmax(dim=1)
    match21 = S.argmax(dim=0)
    N0 = S.shape[0]
    idx0 = torch.arange(N0, device=S.device)
    mutual = match21[match12] == idx0
    M = torch.zeros_like(S)
    M[idx0[mutual], match12[mutual]] = 1.0
    return M

def soft_mnn(f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
    S = f0 @ f1.t()
    P12 = torch.softmax(S, dim=-1)
    P21 = torch.softmax(S.t(), dim=-1)
    return P12 * P21.t()

def soft_mnn_with_tau(f0: torch.Tensor, 
                      f1: torch.Tensor, 
                      tau: float = 0.2) -> torch.Tensor:
    S = f0 @ f1.t()
    P12 = torch.softmax(S / tau, dim=-1)
    P21 = torch.softmax(S.t() / tau, dim=-1)
    return P12 * P21.t()

def topk_soft_mnn_with_tau(f0: torch.Tensor, 
                           f1: torch.Tensor, 
                           tau: float = 0.2, 
                           k: int = 50) -> torch.Tensor:
    S = f0 @ f1.t()
    # 行方向
    v,i = S.topk(k, dim=-1)
    row = torch.full_like(S, float('-inf'))
    row.scatter_(1, i, v)
    P12 = torch.softmax(row / tau, dim=-1)
    # 列方向
    v2,i2 = S.t().topk(k, dim=-1)
    col = torch.full_like(S.t(), float('-inf'))
    col.scatter_(1, i2, v2)
    P21 = torch.softmax(col / tau, dim=-1)
    return P12 * P21.t() 

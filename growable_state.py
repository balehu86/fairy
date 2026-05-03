# growable_state.py — 加投影集中正则

import torch
import torch.nn as nn
import torch.nn.functional as F


class GrowableStateSpace(nn.Module):
    def __init__(self, vec_dim=32, query_dim=64, max_init=4, dist_threshold=1.5):
        super().__init__()
        self.vec_dim = vec_dim
        self.dist_threshold = dist_threshold
        self.query_proj = nn.Linear(query_dim, vec_dim)
        self.update_gate = nn.Linear(vec_dim, 1)
        self.seed = nn.Parameter(torch.randn(max_init, vec_dim) * 0.02)
        self.new_entry_proj = nn.Linear(query_dim, vec_dim)
    
    def read(self, query, pool):
        q = self.query_proj(query)
        if pool.shape[0] == 0:
            return torch.zeros(self.vec_dim, device=query.device)
        attn = F.softmax(pool @ q / (self.vec_dim ** 0.5), dim=0)
        return (attn.unsqueeze(-1) * pool).sum(0)

    def write(self, query, pool, threshold=None):
        q = self.query_proj(query)
        new_vec = self.new_entry_proj(query)
        thresh = threshold if threshold is not None else self.dist_threshold
        
        if pool.shape[0] == 0:
            return new_vec.unsqueeze(0)
        
        dist = (pool - q.unsqueeze(0)).norm(dim=-1)
        best_idx = dist.argmin()
        
        if dist[best_idx].item() < thresh:
            gate = torch.sigmoid(self.update_gate(pool[best_idx] - q))
            updated = pool.clone()
            updated[best_idx] = (1 - gate) * pool[best_idx] + gate * new_vec
            return updated.detach()
        else:
            return torch.cat([pool.detach(), new_vec.unsqueeze(0)], dim=0)
    
    def init_pool(self, device):
        return self.seed.data.clone().to(device)
    
    def concentration_loss(self, query, pool):
        """让query_proj输出更集中: 相似输入应投影到相近位置"""
        if pool.shape[0] < 2:
            return torch.tensor(0.0, device=query.device)
        q = self.query_proj(query)
        # 鼓励query靠近最近的pool条目
        dist = (pool - q.unsqueeze(0)).norm(dim=-1)
        min_dist = dist.min()
        return min_dist * 0.1  # 拉近query到最近的concept
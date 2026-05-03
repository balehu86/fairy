# growable_state.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GrowableStateSpace(nn.Module):
    """可扩充概念状态池: 读=注意力(固定输出), 写=匹配则更新/不匹配则追加"""
    def __init__(self, vec_dim=32, query_dim=64, max_init=4):
        super().__init__()
        self.vec_dim = vec_dim
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

    def write(self, query, pool, threshold=0.3):
        q = self.query_proj(query)
        new_vec = self.new_entry_proj(query)
        if pool.shape[0] == 0:
            return new_vec.unsqueeze(0)
        sim = F.cosine_similarity(pool, q.unsqueeze(0), dim=-1)
        best_idx = sim.argmax()
        if sim[best_idx].item() > threshold:
            gate = torch.sigmoid(self.update_gate(pool[best_idx] - q))
            updated = pool.clone()
            updated[best_idx] = (1 - gate) * pool[best_idx] + gate * new_vec
            return updated.detach()  # 断开历史图, 防止80步累积爆炸
        else:
            return torch.cat([pool.detach(), new_vec.unsqueeze(0)], dim=0)
    
    def init_pool(self, device):
        return self.seed.data.clone().to(device)
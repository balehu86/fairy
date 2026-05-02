import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============ 1. 编码器 ============
class Encoder(nn.Module):
    def __init__(self, obs_dim=56, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
    def forward(self, x): return self.net(x)

# ============ 2. C₋₁ 元目标层 ============
class MetaGoalLayer(nn.Module):
    def __init__(self, input_dim=64, goal_dim=8):
        super().__init__()
        self.priority_head = nn.Linear(input_dim + 3, goal_dim)
        self.goal_dim = goal_dim
    
    def forward(self, h, curiosity, capability_gap, ext_reward):
        signals = torch.tensor([curiosity, capability_gap, ext_reward], 
                               dtype=torch.float32)
        combined = torch.cat([h, signals])
        goal = torch.tanh(self.priority_head(combined))
        return goal

# ============ 3. 场景感知稀疏路由 ============
class SparseSceneRouter(nn.Module):
    def __init__(self, input_dim=64, n_scenes=16, activation_rate=0.15):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_scenes, input_dim) * 0.02)
        self.goal_modulator = nn.Linear(8, n_scenes)
        self.n_scenes = n_scenes
        self.activation_rate = activation_rate
        self.masks = nn.Parameter(torch.randn(n_scenes, input_dim) * 0.02)
    
    def forward(self, h, goal, tau=1.0, hard=False):
        sim = F.cosine_similarity(h.unsqueeze(0), self.prototypes, dim=-1)
        goal_bias = self.goal_modulator(goal)
        logits = sim + goal_bias
        
        if hard:
            weights = F.gumbel_softmax(logits, tau=tau, hard=True)
        else:
            weights = F.softmax(logits / tau, dim=-1)
        
        combined_mask = torch.sigmoid((weights.unsqueeze(-1) * self.masks).sum(0))
        k = max(1, int(self.activation_rate * combined_mask.shape[0]))
        topk_val, topk_idx = combined_mask.topk(k)
        sparse_mask = torch.zeros_like(combined_mask)
        sparse_mask.scatter_(0, topk_idx, topk_val)
        
        return h * sparse_mask, weights

# ============ 4. 门控 SSM (稳定版 Mamba 近似) ============
class GatedSSM(nn.Module):
    """
    用 GRU 门控替代裸 A 矩阵递归。
    这是 SSM 的稳定离散化——Mamba 本质上也是做这件事。
    """
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.state_dim = state_dim
        # 门控参数
        self.W_z = nn.Linear(input_dim + state_dim, state_dim)  # 遗忘门
        self.W_r = nn.Linear(input_dim + state_dim, state_dim)  # 重置门
        self.W_h = nn.Linear(input_dim + state_dim, state_dim)  # 候选状态
        # ΔA 调制接口: 修改重置门的偏置
        self.delta_bias_proj = nn.Linear(state_dim, state_dim)
    
    def forward(self, x, S, delta_A_signal=None):
        combined = torch.cat([x, S], dim=-1)
        z = torch.sigmoid(self.W_z(combined))    # 遗忘门
        r = torch.sigmoid(self.W_r(combined))    # 重置门
        
        # ΔA 调制: 影响重置门偏置 (而非直接改 A 矩阵)
        r_bias = 0
        if delta_A_signal is not None:
            r_bias = self.delta_bias_proj(delta_A_signal) * 0.1
        r = torch.clamp(r + r_bias, 0, 1)
        
        combined_r = torch.cat([x, r * S], dim=-1)
        h_hat = torch.tanh(self.W_h(combined_r))  # 候选状态
        
        S_new = (1 - z) * S + z * h_hat
        return S_new, S_new  # 输出 = 新状态 (无额外输出投影, 简化)

# ============ 5. 元认知层 S_meta ============
class MetaCognition(nn.Module):
    def __init__(self, obj_state_dim=32, meta_state_dim=16):
        super().__init__()
        trace_dim = obj_state_dim + 3
        self.trace_encoder = nn.Linear(trace_dim, meta_state_dim)
        self.meta_ssm = GatedSSM(meta_state_dim, meta_state_dim)
        self.delta_signal_proj = nn.Linear(meta_state_dim, obj_state_dim)
        self.interp_head = nn.Linear(meta_state_dim, 4)
        self.obj_dim = obj_state_dim
    
    def forward(self, S_obj, pred_err, confidence, entropy, S_meta):
        trace = torch.cat([S_obj, 
                           torch.tensor([pred_err, confidence, entropy])])
        x = self.trace_encoder(trace)
        S_meta_new, _ = self.meta_ssm(x, S_meta)
        
        # 生成调制信号 (注意: 不再是低秩A矩阵, 而是偏置调制)
        delta_signal = torch.tanh(self.delta_signal_proj(S_meta_new))
        interp = torch.sigmoid(self.interp_head(S_meta_new))
        
        return delta_signal, S_meta_new, interp

# ============ 6. 槽位工作记忆 ============
class SlotMemory(nn.Module):
    def __init__(self, slot_dim=64, n_slots=4):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.write_gate = nn.Linear(slot_dim, n_slots)
        self.read_attn = nn.Linear(slot_dim, slot_dim)
        self.empty_token = nn.Parameter(torch.zeros(slot_dim))
    
    def init_slots(self):
        return self.empty_token.unsqueeze(0).expand(self.n_slots, -1).clone()
    
    def forward(self, query, slots):
        q = self.read_attn(query)
        attn = F.softmax(slots @ q / (self.slot_dim**0.5), dim=0)
        read = (attn.unsqueeze(-1) * slots).sum(0)
        
        write_logits = self.write_gate(query)
        write_weights = F.softmax(write_logits, dim=0)
        new_slots = slots + write_weights.unsqueeze(-1) * (query - slots) * 0.3
        
        return read, new_slots

# ============ 7. 因果图模块 ============
class CausalGraph(nn.Module):
    def __init__(self, n_vars=4, state_dim=64):
        super().__init__()
        self.n_vars = n_vars
        self.var_detector = nn.Linear(state_dim, n_vars)
        self.adj_logits = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        self.struct_eq = nn.ModuleList([
            nn.Linear(n_vars, 1) for _ in range(n_vars)
        ])
    
    def detect_vars(self, h):
        return torch.sigmoid(self.var_detector(h))
    
    def get_adj(self):
        adj = torch.sigmoid(self.adj_logits)
        adj = adj * (1 - torch.eye(self.n_vars))
        return adj
    
    def predict(self, var_probs):
        adj = self.get_adj()
        preds = []
        for i in range(self.n_vars):
            parents = var_probs * adj[:, i]
            preds.append(torch.sigmoid(self.struct_eq[i](parents)))
        return torch.cat(preds)
    
    def counterfactual(self, var_probs, intervene_idx, intervene_val):
        modified = var_probs.clone()
        modified[intervene_idx] = intervene_val
        adj = self.get_adj().clone()
        adj[:, intervene_idx] = 0
        preds = []
        for i in range(self.n_vars):
            parents = modified * adj[:, i]
            preds.append(torch.sigmoid(self.struct_eq[i](parents)))
        return torch.cat(preds)
    
    def sparsity_loss(self):
        return self.get_adj().abs().sum()

# ============ 8. 动作头 ============
class ActionHead(nn.Module):
    def __init__(self, input_dim, n_actions=6):
        super().__init__()
        self.policy = nn.Linear(input_dim, n_actions)
        self.value = nn.Linear(input_dim, 1)
    
    def forward(self, h):
        return F.softmax(self.policy(h), dim=-1), self.value(h)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, obs_dim=56, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
    def forward(self, x): return self.net(x)

class MetaGoalLayer(nn.Module):
    def __init__(self, input_dim=64, goal_dim=8):
        super().__init__()
        self.priority_head = nn.Linear(input_dim + 3, goal_dim)
    def forward(self, h, curiosity, capability_gap, ext_reward):
        signals = torch.tensor([curiosity, capability_gap, ext_reward], dtype=torch.float32)
        return torch.tanh(self.priority_head(torch.cat([h, signals])))

class SparseSceneRouter(nn.Module):
    def __init__(self, input_dim=64, n_scenes=16):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_scenes, input_dim) * 0.02)
        self.goal_modulator = nn.Linear(8, n_scenes)
        self.enhancements = nn.Parameter(torch.zeros(n_scenes, input_dim))
        self.gate = nn.Linear(input_dim + 8, 1)
    def forward(self, h, goal, tau=1.0):
        sim = F.cosine_similarity(h.unsqueeze(0), self.prototypes, dim=-1)
        goal_bias = self.goal_modulator(goal)
        weights = F.softmax((sim + goal_bias) / tau, dim=-1)
        enhancement = (weights.unsqueeze(-1) * self.enhancements).sum(0)
        alpha = torch.sigmoid(self.gate(torch.cat([h, goal]))).squeeze()
        return h + alpha * enhancement, weights

class GatedSSM(nn.Module):
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.W_z = nn.Linear(input_dim + state_dim, state_dim)
        self.W_r = nn.Linear(input_dim + state_dim, state_dim)
        self.W_h = nn.Linear(input_dim + state_dim, state_dim)
        self.output_proj = nn.Linear(state_dim, input_dim)
        self.delta_bias_proj = nn.Linear(state_dim, state_dim)
    def forward(self, x, S, delta_A_signal=None):
        combined = torch.cat([x, S], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        if delta_A_signal is not None:
            r = torch.clamp(r + self.delta_bias_proj(delta_A_signal) * 0.1, 0, 1)
        h_hat = torch.tanh(self.W_h(torch.cat([x, r * S], dim=-1)))
        S_new = (1 - z) * S + z * h_hat
        return self.output_proj(S_new), S_new

class MetaCognition(nn.Module):
    """输入: delta_S_obj(32) + action_repeat_count(1) + 3 scalars = 36"""
    def __init__(self, obj_state_dim=32, meta_state_dim=16):
        super().__init__()
        trace_dim = obj_state_dim + 1 + 3  # +1 for action_repeat
        self.trace_encoder = nn.Linear(trace_dim, meta_state_dim)
        self.meta_ssm = GatedSSM(meta_state_dim, meta_state_dim)
        self.delta_signal_proj = nn.Linear(meta_state_dim, obj_state_dim)
        self.interp_head = nn.Linear(meta_state_dim, 4)
    def forward(self, delta_S_obj, action_repeat, pred_err, confidence, entropy, S_meta):
        trace = torch.cat([delta_S_obj, torch.tensor([action_repeat]),
                           torch.tensor([pred_err, confidence, entropy])])
        x = self.trace_encoder(trace)
        _, S_meta_new = self.meta_ssm(x, S_meta)
        delta_signal = torch.tanh(self.delta_signal_proj(S_meta_new))
        interp = torch.sigmoid(self.interp_head(S_meta_new))
        return delta_signal, S_meta_new, interp

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
        write_weights = F.softmax(self.write_gate(query), dim=0)
        new_slots = slots + write_weights.unsqueeze(-1) * (query - slots) * 0.3
        return read, new_slots

class CausalGraph(nn.Module):
    """
    v2.1-final:
    - 变量→变量边 (4x4 A矩阵)
    - 动作→变量边 (6x4 B矩阵，新增!)
    - predict_delta: Delta_v_i = struct_eq(A·parents, B·action)
    - 只在事件步训练转移预测
    """
    def __init__(self, n_vars=4, n_actions=6, state_dim=64):
        super().__init__()
        self.n_vars = n_vars
        self.n_actions = n_actions
        self.var_detector = nn.Linear(state_dim, n_vars)
        # 变量→变量
        self.adj_logits = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        # 动作→变量 (新增! 关键!)
        self.action_adj_logits = nn.Parameter(torch.randn(n_actions, n_vars) * 0.1)
        # 结构方程: parents(4) + action_effect(4) = 8
        self.struct_eq = nn.ModuleList([
            nn.Linear(n_vars + n_vars, 1) for _ in range(n_vars)
        ])
    
    def detect_vars(self, h):
        return torch.sigmoid(self.var_detector(h))
    
    def get_adj(self):
        adj = torch.sigmoid(self.adj_logits) * (1 - torch.eye(self.n_vars))
        return adj
    
    def get_action_adj(self):
        return torch.sigmoid(self.action_adj_logits)
    
    def predict_delta(self, var_probs, action_oh):
        adj = self.get_adj()
        action_adj = self.get_action_adj()
        parent_signal = var_probs @ adj  # (4,) weighted parents
        action_signal = action_oh @ action_adj  # (4,) which actions affect which vars
        deltas = []
        for i in range(self.n_vars):
            x = torch.cat([parent_signal, action_signal])
            deltas.append(torch.tanh(self.struct_eq[i](x)))
        return torch.cat(deltas)
    
    def counterfactual(self, var_probs, intervene_idx, intervene_val):
        modified = var_probs.clone()
        modified[intervene_idx] = intervene_val
        adj = self.get_adj().clone()
        adj[:, intervene_idx] = 0
        preds = []
        for i in range(self.n_vars):
            parents = modified * adj[:, i]
            dummy_action = torch.zeros(self.n_actions)
            action_adj = self.get_action_adj()
            action_signal = dummy_action @ action_adj
            x = torch.cat([parents, action_signal])
            preds.append(torch.sigmoid(self.struct_eq[i](x)))
        return torch.cat(preds)
    
    def sparsity_loss(self):
        return (self.get_adj().abs() + 1e-8).sqrt().sum() + \
               (self.get_action_adj().abs() + 1e-8).sqrt().sum()

class ActionHead(nn.Module):
    def __init__(self, input_dim, n_actions=6):
        super().__init__()
        self.policy = nn.Linear(input_dim, n_actions)
        self.value = nn.Linear(input_dim, 1)
    def forward(self, h):
        return F.softmax(self.policy(h), dim=-1), self.value(h)
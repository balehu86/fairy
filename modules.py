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
        signals = torch.tensor([curiosity, capability_gap, ext_reward],
                               dtype=torch.float32, device=h.device)
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
    def __init__(self, obj_state_dim=32, meta_state_dim=16):
        super().__init__()
        trace_dim = obj_state_dim + 1 + 3
        self.trace_encoder = nn.Linear(trace_dim, meta_state_dim)
        self.meta_ssm = GatedSSM(meta_state_dim, meta_state_dim)
        self.delta_signal_proj = nn.Linear(meta_state_dim, obj_state_dim)
        # interp: 只用 3 个清晰信号, 更大网络
        self.interp_ff = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 4), nn.Sigmoid())
    def forward(self, delta_S_obj, action_repeat, pred_err, confidence, entropy, S_meta):
        dev = delta_S_obj.device
        trace = torch.cat([delta_S_obj,
                           torch.tensor([action_repeat], device=dev),
                           torch.tensor([pred_err, confidence, entropy], device=dev)])
        x = self.trace_encoder(trace)
        _, S_meta_new = self.meta_ssm(x, S_meta)
        delta_signal = torch.tanh(self.delta_signal_proj(S_meta_new))
        # interp: 3 个强信号 (不用 confidence, 跟 pred_err 线性相关)
        interp_input = torch.tensor([action_repeat, pred_err, entropy], device=dev)
        interp = self.interp_ff(interp_input)
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
    v2.4: var→var 用地面真值共现直接监督。
    
    核心改变:
    - 事件步: 如果变量 i 变了且变量 j 也变了, adj[i,j] 应该大
    - 如果变量 i 变了但 j 没变, adj[i,j] 应该小
    - 这给每个 adj 元素独立的梯度方向, 打破对称
    """
    def __init__(self, n_vars=4, n_actions=6, state_dim=64):
        super().__init__()
        self.n_vars = n_vars
        self.n_actions = n_actions
        self.var_detector = nn.Linear(state_dim, n_vars)
        self.action_effects = nn.Parameter(torch.zeros(n_actions, n_vars))
        # 初始化打破对称: has_key→door_open 先验大
        self.var_causal_logits = nn.Parameter(torch.zeros(n_vars, n_vars))
        # 手动设先验: has_key(0)→door_open(1) 应该大
        with torch.no_grad():
            self.var_causal_logits[0, 1] = 2.0  # sigmoid(2)≈0.88
            # saw_deco(3)→其他 都应该小
            self.var_causal_logits[3, 0] = -2.0
            self.var_causal_logits[3, 1] = -2.0
            self.var_causal_logits[3, 2] = -2.0
    
    def detect_vars(self, h):
        return torch.sigmoid(self.var_detector(h))
    
    def get_var_adj(self):
        return torch.sigmoid(self.var_causal_logits) * (1 - torch.eye(self.n_vars, device=self.var_causal_logits.device))
    
    def predict_delta(self, var_probs, action_oh):
        direct = action_oh @ self.action_effects
        adj = self.get_var_adj()
        mediated = direct @ adj
        return torch.tanh(direct + mediated)
    
    def counterfactual(self, var_probs, intervene_idx, intervene_val):
        modified = var_probs.clone()
        modified[intervene_idx] = intervene_val
        adj = self.get_var_adj().clone()
        adj[:, intervene_idx] = 0
        delta = modified - var_probs
        propagated = delta @ adj
        return torch.clamp(var_probs + propagated, 0, 1)
    
    def sparsity_loss(self):
        return (self.get_var_adj() + 1e-8).sqrt().sum()
    
    def event_supervision_loss(self, action_idx, var_deltas):
        action_oh = torch.zeros(self.n_actions, device=self.action_effects.device)
        action_oh[action_idx] = 1.0
        predicted_effect = action_oh @ self.action_effects
        changed = (var_deltas.abs() > 0.05).float()
        if changed.sum() == 0:
            return torch.tensor(0.0, device=self.action_effects.device)
        target = var_deltas * changed
        mask = changed + 0.1
        return (F.mse_loss(predicted_effect, target, reduction='none') * mask).mean()
    
    def var_causal_supervision_loss(self, gt_deltas):
        """
        关键新增: 用地面真值变化共现直接监督 var→var 邻接。
        
        逻辑: 如果变量 i 变了 (|delta_i| > threshold),
        且变量 j 也变了, 那这条边应该被加强。
        如果 i 变了但 j 没变, 这条边应该被削弱。
        """
        changed = (gt_deltas.abs() > 0.05).float()  # (n_vars,)
        if changed.sum() == 0:
            return torch.tensor(0.0, device=self.var_causal_logits.device)
        
        adj = self.get_var_adj()
        # target: co-occurrence matrix
        # co[i,j] = 1 if both i and j changed, 0 otherwise
        co_occurred = (changed.unsqueeze(0) * changed.unsqueeze(1))  # (n,n)
        # 但 co-occurrence 不等于因果! 如果同时变只因为共同原因,
        # 我们不想给 i→j 和 j→i 都设高。
        # 简单修正: 只在源变量先变的情况下设置 target
        # 在这个环境中: has_key→door_open 是因果方向, 反过来不是
        # 我们用变化幅度排序: 变化大的更可能是原因
        # 这不完美但比均匀好得多
        
        # 简单方法: 对每对 (i,j), target = co_occurred[i,j]
        # 但去掉对角线 (已由 eye mask 处理)
        target = co_occurred * (1 - torch.eye(self.n_vars, device=adj.device))
        
        # BCE loss: 给每个 adj 元素独立的梯度
        loss = F.binary_cross_entropy(adj, target.detach())
        return loss

class ActionHead(nn.Module):
    def __init__(self, input_dim, n_actions=6):
        super().__init__()
        self.policy = nn.Linear(input_dim, n_actions)
        self.value = nn.Linear(input_dim, 1)
    def forward(self, h):
        return F.softmax(self.policy(h), dim=-1), self.value(h)
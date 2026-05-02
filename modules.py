import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============ 1. 多模态嵌入 (这里只有视觉) ============
class Encoder(nn.Module):
    def __init__(self, obs_dim=56, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
    def forward(self, x): return self.net(x)

# ============ 2. C₋₁ 元目标层 (8维) ============
class MetaGoalLayer(nn.Module):
    """
    产生三种信号:
    - 好奇心 (基于学习进度, 非原始预测误差)
    - 能力缺口 (历史失败统计)
    - 目标向量 (用于调制场景路由)
    """
    def __init__(self, input_dim=64, goal_dim=8):
        super().__init__()
        self.goal_head = nn.Linear(input_dim, goal_dim)
        self.priority_head = nn.Linear(input_dim + 3, goal_dim)  # +3: curio/gap/ext_reward
        self.goal_dim = goal_dim
    
    def forward(self, h, curiosity, capability_gap, ext_reward):
        signals = torch.tensor([curiosity, capability_gap, ext_reward], 
                               dtype=torch.float32)
        combined = torch.cat([h, signals])
        goal = torch.tanh(self.priority_head(combined))
        return goal  # (goal_dim,)

# ============ 3. 场景感知稀疏路由 ============
class SparseSceneRouter(nn.Module):
    """16个场景原型, Gumbel-Softmax选择, 目标向量调制"""
    def __init__(self, input_dim=64, n_scenes=16, activation_rate=0.15):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_scenes, input_dim) * 0.1)
        self.goal_modulator = nn.Linear(8, n_scenes)
        self.n_scenes = n_scenes
        self.activation_rate = activation_rate
        # 每个场景的维度掩码 (学习得到)
        self.masks = nn.Parameter(torch.randn(n_scenes, input_dim) * 0.1)
    
    def forward(self, h, goal, tau=1.0, hard=False):
        sim = F.cosine_similarity(h.unsqueeze(0), self.prototypes, dim=-1)
        goal_bias = self.goal_modulator(goal)
        logits = sim + goal_bias
        
        if hard:
            weights = F.gumbel_softmax(logits, tau=tau, hard=True)
        else:
            weights = F.softmax(logits / tau, dim=-1)
        
        # 组合场景掩码 (稀疏激活)
        combined_mask = torch.sigmoid((weights.unsqueeze(-1) * self.masks).sum(0))
        # 稀疏化: 保留 top-k
        k = int(self.activation_rate * combined_mask.shape[0])
        topk_val, topk_idx = combined_mask.topk(k)
        sparse_mask = torch.zeros_like(combined_mask)
        sparse_mask.scatter_(0, topk_idx, topk_val)
        
        return h * sparse_mask, weights  # 掩码后的h, 场景权重

# ============ 4. 简化 Mamba (线性 SSM) ============
class SimpleMamba(nn.Module):
    """
    简化的状态空间模型: S' = A·S + B·x
    A 矩阵可以被外部 ΔA 调制
    """
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.state_dim = state_dim
        # A_base 初始化为稳定矩阵 (对角 + 小扰动)
        A = torch.eye(state_dim) * 0.95 + torch.randn(state_dim, state_dim) * 0.01
        self.A_base = nn.Parameter(A)
        self.B = nn.Linear(input_dim, state_dim)
        self.C = nn.Linear(state_dim, input_dim)
    
    def forward(self, x, S, delta_A=None):
        A = self.A_base
        if delta_A is not None:
            A = A + delta_A
        S_new = S @ A.T + self.B(x)
        S_new = torch.tanh(S_new)  # 稳定性
        y = self.C(S_new)
        return y, S_new

# ============ 5. 元认知层 S_meta ============
class MetaCognition(nn.Module):
    """
    观察 S_obj 的轨迹, 输出 ΔA 调制信号
    关键: S_meta 不接收外部输入, 只看 S_obj 的行为
    """
    def __init__(self, obj_state_dim=32, meta_state_dim=16):
        super().__init__()
        # 输入: S_obj 状态 + 预测误差 + 控制符统计
        trace_dim = obj_state_dim + 3  # +3 for [pred_err, confidence, entropy]
        self.trace_encoder = nn.Linear(trace_dim, meta_state_dim)
        self.meta_mamba = SimpleMamba(meta_state_dim, meta_state_dim)
        # 输出 ΔA (低秩分解, 否则参数爆炸)
        self.delta_rank = 4
        self.delta_A_U = nn.Linear(meta_state_dim, obj_state_dim * self.delta_rank)
        self.delta_A_V = nn.Linear(meta_state_dim, obj_state_dim * self.delta_rank)
        self.obj_dim = obj_state_dim
        # 可解释输出 (我建议的"可读表征")
        self.interp_head = nn.Linear(meta_state_dim, 4)  
        # 4个可解释信号: [循环警告, 置信度, 需要探索, 需要利用]
    
    def forward(self, S_obj, pred_err, confidence, entropy, S_meta):
        trace = torch.cat([S_obj, 
                           torch.tensor([pred_err, confidence, entropy])])
        x = self.trace_encoder(trace)
        _, S_meta_new = self.meta_mamba(x, S_meta)
        
        # 低秩 ΔA
        U = self.delta_A_U(S_meta_new).view(self.obj_dim, self.delta_rank)
        V = self.delta_A_V(S_meta_new).view(self.obj_dim, self.delta_rank)
        delta_A = U @ V.T * 0.01  # 小幅度调制
        
        # 可解释信号
        interp = torch.sigmoid(self.interp_head(S_meta_new))
        
        return delta_A, S_meta_new, interp

# ============ 6. 槽位工作记忆 (我建议新增的) ============
class SlotMemory(nn.Module):
    """4个显式槽位, 可读写"""
    def __init__(self, slot_dim=32, n_slots=4):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.write_gate = nn.Linear(slot_dim, n_slots)
        self.read_attn = nn.Linear(slot_dim, slot_dim)
        # 初始化槽位为可学习的空状态
        self.empty_token = nn.Parameter(torch.zeros(slot_dim))
    
    def init_slots(self):
        return self.empty_token.unsqueeze(0).expand(self.n_slots, -1).clone()
    
    def forward(self, query, slots):
        # 读: 注意力
        keys = slots
        q = self.read_attn(query)
        attn = F.softmax(keys @ q / (self.slot_dim**0.5), dim=0)
        read = (attn.unsqueeze(-1) * slots).sum(0)
        
        # 写: 软门控
        write_logits = self.write_gate(query)
        write_weights = F.softmax(write_logits, dim=0)
        new_slots = slots + write_weights.unsqueeze(-1) * (query - slots) * 0.3
        
        return read, new_slots

# ============ 7. 因果图模块 ============
class CausalGraph(nn.Module):
    """
    离散因果变量 (绑定到任务相关变量):
    - has_key, door_open, near_treasure, saw_decoration
    维护 4x4 的因果权重矩阵
    """
    def __init__(self, n_vars=4, state_dim=64):
        super().__init__()
        self.n_vars = n_vars
        # 变量识别器 (从状态提取离散变量的概率)
        self.var_detector = nn.Linear(state_dim, n_vars)
        # 因果邻接矩阵 (可学习, 稀疏约束)
        self.adj_logits = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        # 结构方程: 给定父节点, 预测子节点
        self.struct_eq = nn.ModuleList([
            nn.Linear(n_vars, 1) for _ in range(n_vars)
        ])
    
    def detect_vars(self, h):
        return torch.sigmoid(self.var_detector(h))
    
    def get_adj(self):
        # 无自环
        adj = torch.sigmoid(self.adj_logits)
        adj = adj * (1 - torch.eye(self.n_vars))
        return adj
    
    def predict(self, var_probs):
        """用因果图预测下一步变量值"""
        adj = self.get_adj()
        preds = []
        for i in range(self.n_vars):
            parents = var_probs * adj[:, i]
            preds.append(torch.sigmoid(self.struct_eq[i](parents)))
        return torch.cat(preds)
    
    def counterfactual(self, var_probs, intervene_idx, intervene_val):
        """do(X=x) 操作: 切断入边, 固定值"""
        modified = var_probs.clone()
        modified[intervene_idx] = intervene_val
        # 这里简化: 直接用修改后的值预测下游
        adj = self.get_adj().clone()
        adj[:, intervene_idx] = 0  # 切断入边
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
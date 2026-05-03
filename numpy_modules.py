# numpy_modules.py — 纯NumPy实现, 无需PyTorch

import numpy as np


class Module:
    """极简模块基类, 手动追踪参数"""
    def parameters(self):
        params = []
        for name, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                params.append((self, name, val))
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params


def kaiming_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

def zeros_init(*shape):
    return np.zeros(shape)


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        self.weight = kaiming_init(in_dim, out_dim)
        self.bias = np.zeros(out_dim)
        self.gw = np.zeros_like(self.weight)
        self.gb = np.zeros_like(self.bias)
        # 缓存前向值用于反向传播
        self._input = None
    
    def forward(self, x):
        self._input = x.copy()
        return x @ self.weight + self.bias
    
    def backward(self, grad_output):
        self.gw += self._input.reshape(-1, 1) * grad_output.reshape(1, -1)
        self.gb += grad_output
        return grad_output @ self.weight.T


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class GatedSSM(Module):
    def __init__(self, input_dim, state_dim):
        self.W_z = Linear(input_dim + state_dim, state_dim)
        self.W_r = Linear(input_dim + state_dim, state_dim)
        self.W_h = Linear(input_dim + state_dim, state_dim)
        self.output_proj = Linear(state_dim, input_dim)
    
    def forward(self, x, S, delta_A=None):
        combined = np.concatenate([x, S])
        z = _sigmoid(self.W_z.forward(combined))
        r = _sigmoid(self.W_r.forward(combined))
        if delta_A is not None:
            r = np.clip(r + delta_A * 0.3, 0, 1)
        r_S = r * S
        h_hat = np.tanh(self.W_h.forward(np.concatenate([x, r_S])))
        S_new = (1 - z) * S + z * h_hat
        if delta_A is not None:
            S_new = S_new + delta_A * 0.1
        y = self.output_proj.forward(S_new)
        return y, S_new


class LowRankDeltaA(Module):
    def __init__(self, meta_dim=16, state_dim=32, rank=4):
        self.down = Linear(meta_dim, rank)
        self.up = Linear(rank, state_dim)
        self.norm_gate = Linear(meta_dim, 1)
        self.up.weight *= 0.01
        self.up.bias *= 0.0
    
    def forward(self, S_meta):
        delta = self.up.forward(_relu(self.down.forward(S_meta)))
        scale = _sigmoid(self.norm_gate.forward(S_meta))
        return delta * scale


class Encoder(Module):
    def __init__(self, obs_dim=56, hidden=64):
        self.net = Sequential(
            Linear(obs_dim, hidden), 
            Lambda(_gelu), 
            Linear(hidden, hidden))
    
    def forward(self, x):
        return self.net.forward(x)


class Lambda:
    def __init__(self, fn):
        self.fn = fn
    def forward(self, x):
        return self.fn(x)


class MetaGoalLayer(Module):
    def __init__(self, input_dim=64, goal_dim=8):
        self.head = Linear(input_dim + 3, goal_dim)
    
    def forward(self, h, curiosity, capability_gap, ext_reward):
        signals = np.array([curiosity, capability_gap, ext_reward])
        return np.tanh(self.head.forward(np.concatenate([h, signals])))


class SparseSceneRouter(Module):
    def __init__(self, input_dim=64, n_scenes=16):
        self.prototypes = np.random.randn(n_scenes, input_dim) * 0.02
        self.goal_modulator = Linear(8, n_scenes)
        self.enhancements = np.zeros((n_scenes, input_dim))
        self.gate = Linear(input_dim + 8, 1)
    
    def forward(self, h, goal, tau=1.0):
        sim = np.array([np.dot(h, p) / (np.linalg.norm(h) * np.linalg.norm(p) + 1e-8) 
                        for p in self.prototypes])
        goal_bias = self.goal_modulator.forward(goal)
        weights = _softmax((sim + goal_bias) / tau)
        enhancement = (weights.reshape(-1, 1) * self.enhancements).sum(0)
        alpha = _sigmoid(self.gate.forward(np.concatenate([h, goal]))).squeeze()
        return h + alpha * enhancement, weights


class MetaCognition(Module):
    def __init__(self, obj_state_dim=32, meta_state_dim=16, delta_rank=4):
        self.trace_encoder = Linear(obj_state_dim + 1 + 3, meta_state_dim)
        self.meta_ssm = GatedSSM(meta_state_dim, meta_state_dim)
        self.low_rank_delta_a = LowRankDeltaA(meta_state_dim, obj_state_dim, delta_rank)
        self.loop_head = Sequential(Linear(1, 8), Lambda(_relu), Linear(8, 1), Lambda(_sigmoid))
        self.interp_shared = Sequential(Linear(3, 16), Lambda(_relu), Linear(16, 3), Lambda(_sigmoid))
    
    def forward(self, delta_S_obj, action_repeat, pred_err, entropy, ext_reward, S_meta):
        trace = np.concatenate([delta_S_obj, [action_repeat], [pred_err, entropy, ext_reward]])
        x = self.trace_encoder.forward(trace)
        _, S_meta_new = self.meta_ssm.forward(x, S_meta)
        delta_A = self.low_rank_delta_a.forward(S_meta_new)
        
        loop = self.loop_head.forward(np.array([action_repeat]))
        shared = self.interp_shared.forward(np.array([pred_err, entropy, ext_reward]))
        interp = np.concatenate([loop, shared])
        
        return delta_A, S_meta_new, interp


class SlotMemory(Module):
    def __init__(self, slot_dim=64, n_slots=4):
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.write_gate = Linear(slot_dim, n_slots)
        self.read_attn = Linear(slot_dim, slot_dim)
        self.empty_token = np.zeros(slot_dim)
    
    def init_slots(self):
        return np.tile(self.empty_token, (self.n_slots, 1)).copy()
    
    def forward(self, query, slots):
        q = self.read_attn.forward(query)
        attn = _softmax(slots @ q / (self.slot_dim**0.5))
        read = (attn.reshape(-1, 1) * slots).sum(0)
        write_weights = _softmax(self.write_gate.forward(query))
        new_slots = slots + write_weights.reshape(-1, 1) * (query - slots) * 0.3
        return read, new_slots


class CausalGraph(Module):
    def __init__(self, n_vars=4, n_actions=6, state_dim=64):
        self.n_vars = n_vars
        self.n_actions = n_actions
        self.var_detector = Linear(state_dim, n_vars)
        self.action_effects = np.zeros((n_actions, n_vars))
        self.var_causal_logits = np.zeros((n_vars, n_vars))
        self.var_causal_logits[0, 1] = 2.0
        self.var_causal_logits[3, 0] = -2.0
        self.var_causal_logits[3, 1] = -2.0
        self.var_causal_logits[3, 2] = -2.0
    
    def detect_vars(self, h):
        return _sigmoid(self.var_detector.forward(h))
    
    def get_var_adj(self):
        adj = _sigmoid(self.var_causal_logits)
        adj *= (1 - np.eye(self.n_vars))
        return adj
    
    def counterfactual(self, var_probs, intervene_idx, intervene_val):
        modified = var_probs.copy()
        modified[intervene_idx] = intervene_val
        adj = self.get_var_adj().copy()
        adj[:, intervene_idx] = 0
        delta = modified - var_probs
        propagated = delta @ adj
        return np.clip(var_probs + propagated, 0, 1)
    
    def sparsity_loss_val(self):
        adj = self.get_var_adj()
        return np.sqrt(adj + 1e-8).sum()
    
    def event_supervision_loss_val(self, action_idx, var_deltas):
        action_oh = np.zeros(self.n_actions)
        action_oh[action_idx] = 1.0
        predicted = action_oh @ self.action_effects
        changed = (np.abs(var_deltas) > 0.05).astype(float)
        if changed.sum() == 0:
            return 0.0
        target = var_deltas * changed
        mask = changed + 0.1
        return (np.sum((predicted - target)**2 * mask) / mask.sum())
    
    def var_causal_supervision_loss_val(self, gt_deltas):
        changed = (np.abs(gt_deltas) > 0.05).astype(float)
        if changed.sum() == 0:
            return 0.0
        adj = self.get_var_adj()
        co_occurred = changed.reshape(1, -1) * changed.reshape(-1, 1)
        target = co_occurred * (1 - np.eye(self.n_vars))
        eps = 1e-7
        adj_c = np.clip(adj, eps, 1 - eps)
        bce = -target * np.log(adj_c) - (1 - target) * np.log(1 - adj_c)
        return bce.sum() / (self.n_vars * (self.n_vars - 1))


class ActionHead(Module):
    def __init__(self, input_dim, n_actions=6):
        self.policy = Linear(input_dim, n_actions)
        self.value_head = Linear(input_dim, 1)
    
    def forward(self, h):
        probs = _softmax(self.policy.forward(h))
        value = self.value_head.forward(h)
        return probs, value


# ── 激活函数 ──

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _relu(x):
    return np.maximum(0, x)

def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def _softmax(x):
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-8)
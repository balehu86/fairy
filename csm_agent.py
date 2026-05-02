import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from modules import *

class CSMv2Agent(nn.Module):
    def __init__(self, obs_dim=56, n_actions=6):
        super().__init__()
        H = 64
        
        self.encoder = Encoder(obs_dim, H)
        self.meta_goal = MetaGoalLayer(H, goal_dim=8)
        self.scene_router = SparseSceneRouter(H, n_scenes=16, activation_rate=0.15)
        
        self.obj_ssm = GatedSSM(H, state_dim=32)
        self.meta_cog = MetaCognition(obj_state_dim=32, meta_state_dim=16)
        self.slot_mem = SlotMemory(slot_dim=H, n_slots=4)
        self.causal_graph = CausalGraph(n_vars=4, state_dim=H)
        
        self.c0_proj = nn.Linear(H, 16)
        self.c1_proj = nn.Linear(32, 32)
        self.c2_proj = nn.Linear(32 + H, 64)
        
        self.action_head = ActionHead(64, n_actions)
        
        # 世界模型 (辅助损失用)
        self.world_model = nn.Linear(H + n_actions, H)
        
        self.pred_err_history = deque(maxlen=200)
        self.global_step = 0
        self.meta_warmup = 800
        self.reset_hidden()
    
    def reset_hidden(self):
        self.S_obj = torch.zeros(32)
        self.S_meta = torch.zeros(16)
        self.slots = self.slot_mem.init_slots()
        self.prev_h = None
        self.prev_action_oh = None
        self._prev_var_probs = None
    
    def compute_learning_progress(self):
        if len(self.pred_err_history) < 20:
            return 0.0
        recent = np.mean(list(self.pred_err_history)[-10:])
        older = np.mean(list(self.pred_err_history)[-20:-10])
        return max(0, older - recent)
    
    def forward(self, obs, ext_reward=0.0):
        self.global_step += 1
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        h = self.encoder(obs_t)
        
        # 世界模型预测误差
        pred_err = 0.0
        if self.prev_h is not None and self.prev_action_oh is not None:
            pred_h = self.world_model(torch.cat([self.prev_h, self.prev_action_oh]))
            pred_err = F.mse_loss(pred_h, h.detach()).item()
            self.pred_err_history.append(pred_err)
        
        curiosity = self.compute_learning_progress()
        capability_gap = max(0, -ext_reward) if ext_reward < 0 else 0.1
        
        goal = self.meta_goal(h.detach(), curiosity, capability_gap, ext_reward)
        h_routed, scene_weights = self.scene_router(h, goal)
        
        confidence = 1.0 - min(pred_err, 1.0)
        entropy = -(scene_weights * torch.log(scene_weights + 1e-8)).sum().item()
        delta_signal, self.S_meta, interp = self.meta_cog(
            self.S_obj.detach(), pred_err, confidence, entropy, self.S_meta
        )
        
        meta_strength = min(1.0, self.global_step / self.meta_warmup)
        delta_signal = delta_signal * meta_strength
        
        self.S_obj, _ = self.obj_ssm(h_routed, self.S_obj, delta_signal)
        slot_read, self.slots = self.slot_mem(h_routed, self.slots)
        
        c0 = self.c0_proj(h)
        c1 = self.c1_proj(self.S_obj)
        c2 = self.c2_proj(torch.cat([self.S_obj, slot_read]))
        
        var_probs = self.causal_graph.detect_vars(h)
        action_probs, value = self.action_head(c2)
        
        self.prev_h = h.detach()
        self._prev_var_probs = var_probs.detach()
        
        return {
            'action_probs': action_probs,
            'value': value,
            'goal': goal,
            'scene_weights': scene_weights,
            'interp': interp,
            'var_probs': var_probs,
            'curiosity': curiosity,
            'pred_err': pred_err,
            'h': h,  # 需要保留给辅助损失
        }
    
    def set_prev_action(self, action, n_actions=6):
        oh = torch.zeros(n_actions)
        oh[action] = 1.0
        self.prev_action_oh = oh
    
    def compute_aux_losses(self, h, prev_h, prev_action_oh, var_probs, prev_var_probs):
        """辅助损失: 给每个模块独立学习信号, 不依赖策略梯度"""
        losses = {}
        
        # 1. 世界模型预测损失
        if prev_h is not None and prev_action_oh is not None:
            pred_h = self.world_model(torch.cat([prev_h, prev_action_oh]))
            losses['world'] = F.mse_loss(pred_h, h.detach())
        
        # 2. 因果图预测损失
        if prev_var_probs is not None:
            target = var_probs.detach()
            pred_vars = self.causal_graph.predict(prev_var_probs.detach())
            losses['causal'] = F.mse_loss(pred_vars, target)
        
        # 3. 因果图稀疏损失
        losses['causal_sparse'] = self.causal_graph.sparsity_loss() * 0.01
        
        return losses
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from modules import *
from device_utils import DEVICE

class CSMv2Agent(nn.Module):
    def __init__(self, obs_dim=56, n_actions=6):
        super().__init__()
        H = 64
        self.n_actions = n_actions
        self.device = DEVICE
        
        self.encoder = Encoder(obs_dim, H).to(DEVICE)
        self.meta_goal = MetaGoalLayer(H, goal_dim=8).to(DEVICE)
        self.scene_router = SparseSceneRouter(H, n_scenes=16).to(DEVICE)
        self.obj_ssm = GatedSSM(H, state_dim=32).to(DEVICE)
        self.meta_cog = MetaCognition(obj_state_dim=32, meta_state_dim=16).to(DEVICE)
        self.slot_mem = SlotMemory(slot_dim=H, n_slots=4).to(DEVICE)
        self.causal_graph = CausalGraph(n_vars=4, n_actions=n_actions, state_dim=H).to(DEVICE)
        
        self.c2_proj = nn.Linear(32 + 64 + 64, 64).to(DEVICE)
        self.action_head = ActionHead(64, n_actions).to(DEVICE)
        self.world_model = nn.Linear(H + n_actions, H).to(DEVICE)
        
        self.pred_err_history = deque(maxlen=200)
        self.global_step = 0
        
        self.phase = 1
        self.phase1_end = 800
        self.phase2_end = 1600
        
        self.reset_hidden()
    
    def reset_hidden(self):
        self.S_obj = torch.zeros(32, device=self.device)
        self.prev_S_obj = torch.zeros(32, device=self.device)
        self.S_meta = torch.zeros(16, device=self.device)
        self.slots = self.slot_mem.init_slots().to(self.device)
        self.prev_h = None
        self.prev_action_oh = None
        self.last_action = -1
        self.action_repeat_count = 0.0
    
    def compute_learning_progress(self):
        if len(self.pred_err_history) < 20: return 0.0
        recent = np.mean(list(self.pred_err_history)[-10:])
        older = np.mean(list(self.pred_err_history)[-20:-10])
        return max(0, older - recent)
    
    def forward(self, obs, ext_reward=0.0, action_taken=None):
        self.global_step += 1
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        h = self.encoder(obs_t)
        
        pred_err = 0.0
        if self.prev_h is not None and self.prev_action_oh is not None:
            pred_h = self.world_model(torch.cat([self.prev_h, self.prev_action_oh]))
            pred_err = F.mse_loss(pred_h, h.detach()).item()
            self.pred_err_history.append(pred_err)
        
        curiosity = self.compute_learning_progress()
        capability_gap = max(0, -ext_reward) if ext_reward < 0 else 0.1
        
        if action_taken is not None:
            if action_taken == self.last_action:
                self.action_repeat_count = min(self.action_repeat_count + 1.0, 20.0)
            else:
                self.action_repeat_count = 0.0
            self.last_action = action_taken
        normed_repeat = self.action_repeat_count / 10.0
        
        goal = self.meta_goal(h.detach(), curiosity, capability_gap, ext_reward)
        if self.phase < 3: goal = goal * 0
        
        h_enhanced, scene_weights = self.scene_router(h, goal)
        
        delta_S_obj = self.S_obj - self.prev_S_obj
        confidence = 1.0 - min(pred_err, 1.0)
        entropy_val = -(scene_weights * torch.log(scene_weights + 1e-8)).sum().item()
        delta_signal, self.S_meta, interp = self.meta_cog(
            delta_S_obj.detach(), normed_repeat, pred_err, confidence, entropy_val, self.S_meta
        )
        if self.phase < 3: delta_signal = delta_signal * 0
        
        y_ssm, self.S_obj = self.obj_ssm(h_enhanced, self.S_obj, delta_signal)
        slot_read, self.slots = self.slot_mem(h_enhanced, self.slots)
        c2 = self.c2_proj(torch.cat([self.S_obj, slot_read, h]))
        
        var_probs = self.causal_graph.detect_vars(h)
        if self.phase < 2: var_probs = var_probs.detach()
        
        action_probs, value = self.action_head(c2)
        
        self.prev_S_obj = self.S_obj.detach().clone()
        self.prev_h = h.detach()
        
        return {
            'action_probs': action_probs, 'value': value,
            'interp': interp, 'var_probs': var_probs,
            'curiosity': curiosity, 'pred_err': pred_err, 'h': h,
        }
    
    def set_prev_action(self, action):
        oh = torch.zeros(self.n_actions, device=self.device)
        oh[action] = 1.0
        self.prev_action_oh = oh
    
    def compute_aux_losses(self, h, prev_h, prev_action_oh, var_probs,
                           prev_var_probs, ground_truth, is_event_step):
        losses = {}
        
        if prev_h is not None and prev_action_oh is not None:
            pred_h = self.world_model(torch.cat([prev_h, prev_action_oh]))
            losses['world'] = F.mse_loss(pred_h, h.detach())
        
        if ground_truth is not None:
            gt = torch.as_tensor(ground_truth, dtype=torch.float32, device=self.device)
            losses['causal_detect'] = F.binary_cross_entropy(var_probs, gt)
        
        if self.phase >= 2 and is_event_step and prev_var_probs is not None and prev_action_oh is not None:
            delta_target = (var_probs - prev_var_probs).detach()
            if delta_target.abs().sum() > 0.01:
                pred_delta = self.causal_graph.predict_delta(
                    prev_var_probs.detach(), prev_action_oh.detach()
                )
                weights = delta_target.abs() + 0.1
                losses['causal_predict'] = (F.mse_loss(pred_delta, delta_target, reduction='none') * weights).mean() * 10.0
                losses['causal_sparse'] = self.causal_graph.sparsity_loss() * 0.5
        
        return losses
    
    def set_phase(self, episode):
        if episode < self.phase1_end: self.phase = 1
        elif episode < self.phase2_end: self.phase = 2
        else: self.phase = 3
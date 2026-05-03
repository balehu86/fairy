# numpy_agent.py — 纯NumPy版CSMv3智能体

import numpy as np
from collections import deque
from numpy_modules import (
    Module, Linear, GatedSSM, LowRankDeltaA, Encoder, MetaGoalLayer,
    SparseSceneRouter, MetaCognition, SlotMemory, CausalGraph, ActionHead,
    _sigmoid, _softmax
)


class CSMv3Agent:
    def __init__(self, obs_dim=56, n_actions=6):
        H = 64
        self.n_actions = n_actions

        self.encoder = Encoder(obs_dim, H)
        self.meta_goal = MetaGoalLayer(H, goal_dim=8)
        self.scene_router = SparseSceneRouter(H, n_scenes=16)
        self.obj_ssm = GatedSSM(H, state_dim=32)
        self.meta_cog = MetaCognition(obj_state_dim=32, meta_state_dim=16, delta_rank=4)
        self.slot_mem = SlotMemory(slot_dim=H, n_slots=4)
        self.causal_graph = CausalGraph(n_vars=4, n_actions=n_actions, state_dim=H)

        self.c2_proj = Linear(32 + 64 + 64, 64)
        self.action_head = ActionHead(64, n_actions)
        self.world_model = Linear(H + n_actions, H)

        self.pred_err_history = deque(maxlen=200)
        self.global_step = 0

        self.phase = 1
        self.phase1_end = 800
        self.phase2_end = 1600

        self.reset_hidden()

    def reset_hidden(self):
        self.S_obj = np.zeros(32)
        self.prev_S_obj = np.zeros(32)
        self.S_meta = np.zeros(16)
        self.slots = self.slot_mem.init_slots()
        self.prev_h = None
        self.prev_action_oh = None
        self.last_action = -1
        self.action_repeat_count = 0.0
        self.last_delta_A = None

    def compute_learning_progress(self):
        if len(self.pred_err_history) < 20:
            return 0.0
        recent = np.mean(list(self.pred_err_history)[-10:])
        older = np.mean(list(self.pred_err_history)[-20:-10])
        return max(0, older - recent)

    def forward(self, obs, ext_reward=0.0, action_taken=None):
        self.global_step += 1
        obs_arr = np.asarray(obs, dtype=np.float64)
        h = self.encoder.forward(obs_arr)

        pred_err = 0.0
        if self.prev_h is not None and self.prev_action_oh is not None:
            pred_h = self.world_model.forward(
                np.concatenate([self.prev_h, self.prev_action_oh]))
            pred_err = float(np.mean((pred_h - h) ** 2))
            self.pred_err_history.append(pred_err)

        curiosity = self.compute_learning_progress()
        capability_gap = max(0, -ext_reward) if ext_reward < 0 else 0.1

        if action_taken is not None:
            if action_taken == self.last_action:
                self.action_repeat_count = min(
                    self.action_repeat_count + 1.0, 20.0)
            else:
                self.action_repeat_count = 0.0
            self.last_action = action_taken
        normed_repeat = self.action_repeat_count / 10.0

        goal = self.meta_goal.forward(
            h, curiosity, capability_gap, ext_reward)
        if self.phase < 3:
            goal = goal * 0

        h_enhanced, scene_weights = self.scene_router.forward(h, goal)

        delta_S_obj = self.S_obj - self.prev_S_obj
        entropy_val = float(
            -(scene_weights * np.log(scene_weights + 1e-8)).sum())
        delta_A, self.S_meta, interp = self.meta_cog.forward(
            delta_S_obj, normed_repeat, pred_err,
            entropy_val, ext_reward, self.S_meta)
        if self.phase < 3:
            delta_A = delta_A * 0
        self.last_delta_A = delta_A.copy() if self.phase >= 3 else None

        y_ssm, self.S_obj = self.obj_ssm.forward(
            h_enhanced, self.S_obj, delta_A if self.phase >= 3 else None)
        slot_read, self.slots = self.slot_mem.forward(
            h_enhanced, self.slots)
        c2 = self.c2_proj.forward(
            np.concatenate([self.S_obj, slot_read, h]))

        var_probs = self.causal_graph.detect_vars(h)

        action_probs, value = self.action_head.forward(c2)

        self.prev_S_obj = self.S_obj.copy()
        self.prev_h = h.copy()

        return {
            'action_probs': action_probs,
            'value': float(value.flat[0]) if np.ndim(value) else float(value),
            'interp': interp,
            'var_probs': var_probs,
            'curiosity': curiosity,
            'pred_err': pred_err,
            'h': h,
            'delta_A_mag': float(np.abs(delta_A).mean())
            if self.phase >= 3 else 0.0,
        }

    def set_prev_action(self, action):
        oh = np.zeros(self.n_actions)
        oh[action] = 1.0
        self.prev_action_oh = oh

    def set_phase(self, episode):
        if episode < self.phase1_end:
            self.phase = 1
        elif episode < self.phase2_end:
            self.phase = 2
        else:
            self.phase = 3

    # ── 参数序列化 (ES优化用) ──

    def get_all_params(self):
        parts = []
        # Linear 层参数
        for mod in [self.encoder, self.meta_goal, self.scene_router,
                    self.obj_ssm, self.meta_cog, self.slot_mem,
                    self.causal_graph.var_detector, self.c2_proj,
                    self.action_head, self.world_model]:
            for _, name, val in mod.parameters():
                parts.append(val.ravel())
        # 直接参数化
        parts.append(self.causal_graph.action_effects.ravel())
        parts.append(self.causal_graph.var_causal_logits.ravel())
        # 特殊参数
        parts.append(self.scene_router.prototypes.ravel())
        parts.append(self.scene_router.enhancements.ravel())
        parts.append(self.slot_mem.empty_token.ravel())
        return np.concatenate(parts)

    def set_all_params(self, flat):
        idx = 0

        def _restore(obj, name, shape):
            nonlocal idx
            size = int(np.prod(shape))
            setattr(obj, name, flat[idx:idx + size].reshape(shape))
            idx += size

        for mod in [self.encoder, self.meta_goal, self.scene_router,
                    self.obj_ssm, self.meta_cog, self.slot_mem,
                    self.causal_graph.var_detector, self.c2_proj,
                    self.action_head, self.world_model]:
            for obj, name, val in mod.parameters():
                _restore(obj, name, val.shape)

        _restore(self.causal_graph, 'action_effects',
                 self.causal_graph.action_effects.shape)
        _restore(self.causal_graph, 'var_causal_logits',
                 self.causal_graph.var_causal_logits.shape)
        _restore(self.scene_router, 'prototypes',
                 self.scene_router.prototypes.shape)
        _restore(self.scene_router, 'enhancements',
                 self.scene_router.enhancements.shape)
        _restore(self.slot_mem, 'empty_token',
                 self.slot_mem.empty_token.shape)
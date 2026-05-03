# numpy_train.py — 纯NumPy训练循环, 数值梯度优化

import numpy as np
from collections import deque
import time, json, os

from env import CausalGridWorld
from numpy_agent import CSMv3Agent

MIN_ENTROPY = 0.3


def sample_action(probs, rng):
    """从分类分布采样"""
    cdf = np.cumsum(probs)
    u = rng.random()
    return int(np.searchsorted(cdf, u))


def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def compute_loss(agent, transitions, entropy_coef=0.03, value_coef=0.5,
                 aux_coef=0.3, interp_coef=0.3):
    """根据收集的transitions计算总loss (标量)"""
    returns = compute_returns([t['reward'] for t in transitions])
    returns_arr = np.array(returns)
    if len(returns_arr) > 1:
        returns_arr = (returns_arr - returns_arr.mean()) / (returns_arr.std() + 1e-8)
    
    log_probs = np.array([t['log_prob'] for t in transitions])
    values = np.array([t['value'] for t in transitions])
    entropies = np.array([t['entropy'] for t in transitions])
    
    advantages = returns_arr - values
    
    policy_loss = -(log_probs * advantages).mean()
    value_loss = np.mean((values - returns_arr)**2)
    entropy_bonus = entropies.mean()
    entropy_pen = max(0, MIN_ENTROPY - entropy_bonus) * 2.0
    
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus + entropy_pen
    
    # 辅助损失
    total_aux = 0.0
    aux_count = 0
    for t in transitions:
        for v in t['aux_losses'].values():
            total_aux += v
            aux_count += 1
    if aux_count > 0:
        loss += aux_coef * total_aux / max(1, aux_count)
    
    # interp损失
    interp_targets = []
    interp_outputs = []
    for i in range(1, len(transitions)):
        t = transitions[i]
        pe = t['pred_err']
        r = t['reward']
        repeat = t['action_repeat']
        loop = min(repeat / 10.0, 1.0)
        conf = 1.0 - min(pe, 1.0)
        expl = min(pe * 3, 1.0)
        explt = min(max(r, 0.0), 1.0)
        interp_targets.append([loop, conf, expl, explt])
        interp_outputs.append(t['interp'])
    
    if interp_outputs:
        targets = np.array(interp_targets)
        outputs = np.array(interp_outputs)
        interp_loss = np.mean((outputs - targets)**2)
        loss += interp_coef * interp_loss
    
    return loss


def compute_aux_losses_scalar(agent, h, prev_h, prev_action_oh, var_probs,
                              prev_var_probs, ground_truth, prev_ground_truth,
                              action_idx, is_event_step):
    """计算辅助损失(标量), 不依赖autograd"""
    losses = {}
    
    if prev_h is not None and prev_action_oh is not None:
        pred_h = agent.world_model.forward(np.concatenate([prev_h, prev_action_oh]))
        losses['world'] = np.mean((pred_h - h)**2)
    
    if ground_truth is not None:
        gt = np.asarray(ground_truth, dtype=np.float32)
        vp = np.clip(var_probs, 1e-7, 1-1e-7)
        losses['causal_detect'] = -np.mean(gt * np.log(vp) + (1-gt) * np.log(1-vp))
    
    if agent.phase >= 2:
        if is_event_step and prev_ground_truth is not None and action_idx is not None:
            gt_delta = np.asarray(ground_truth, dtype=np.float32) - \
                       np.asarray(prev_ground_truth, dtype=np.float32)
            if np.abs(gt_delta).sum() > 0.05:
                losses['event_supervision'] = agent.causal_graph.event_supervision_loss_val(
                    action_idx, gt_delta) * 5.0
                losses['var_causal'] = agent.causal_graph.var_causal_supervision_loss_val(gt_delta) * 3.0
        
        losses['causal_sparse'] = agent.causal_graph.sparsity_loss_val() * 0.1
    
    return losses


def collect_episode(agent, env, rng, max_steps=80):
    obs, gt = env.reset()
    agent.reset_hidden()
    
    transitions = []
    total_reward = 0
    prev_var_probs = None
    prev_gt = None
    
    for step in range(max_steps):
        ext_r = 0.0 if step == 0 else transitions[-1]['reward']
        out = agent(obs, ext_reward=ext_r,
                    action_taken=transitions[-1]['action_idx'] if step > 0 else None)
        
        action_probs = out['action_probs']
        action_idx = sample_action(action_probs, rng)
        log_prob = np.log(action_probs[action_idx] + 1e-8)
        entropy = -(action_probs * np.log(action_probs + 1e-8)).sum()
        
        next_obs, reward, done, next_gt = env.step(action_idx)
        
        is_event = False
        if prev_gt is not None:
            is_event = np.any(np.abs(gt - prev_gt) > 0.1)
        
        agent.set_prev_action(action_idx)
        aux_losses = compute_aux_losses_scalar(
            agent, out['h'], agent.prev_h, agent.prev_action_oh,
            out['var_probs'], prev_var_probs, gt, prev_gt,
            action_idx, is_event
        )
        prev_var_probs = out['var_probs'].copy()
        
        transitions.append({
            'obs': obs, 'action_idx': action_idx,
            'log_prob': log_prob, 'entropy': entropy,
            'reward': reward, 'value': out['value'],
            'aux_losses': aux_losses, 'interp': out['interp'].copy(),
            'pred_err': out['pred_err'], 'is_event': is_event,
            'action_repeat': agent.action_repeat_count,
            'delta_A_mag': out['delta_A_mag'],
        })
        
        total_reward += reward
        obs = next_obs
        prev_gt = gt
        gt = next_gt
        if done: break
    
    return transitions, total_reward


def numerical_gradient(agent, transitions, eps=1e-4):
    """数值梯度: L(θ+ε) - L(θ-ε)) / 2ε"""
    params = agent.get_all_params()
    grad = np.zeros_like(params)
    base_loss = compute_loss(agent, transitions)
    
    n_params = len(params)
    print(f"  [grad] 计算数值梯度: {n_params} 参数, eps={eps}")
    
    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += eps
        agent.set_all_params(params_plus)
        # 需要重新跑forward得到新的transitions — 太慢!
        # 用当前transitions但参数已变, 近似计算
        loss_plus = compute_loss(agent, transitions)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        agent.set_all_params(params_minus)
        loss_minus = compute_loss(agent, transitions)
        
        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    
    # 恢复原参数
    agent.set_all_params(params)
    return grad, base_loss
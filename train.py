# train.py — 修GPU bug + loop BCE加权

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import time, json, os

from env import CausalGridWorld
from csm_agent import CSMv3Agent
from device_utils import DEVICE

MIN_ENTROPY = 0.3


def collect_episode(agent, env, max_steps=80):
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
        
        dist = torch.distributions.Categorical(out['action_probs'])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        next_obs, reward, done, next_gt = env.step(action.item())
        
        is_event = False
        if prev_gt is not None:
            is_event = np.any(np.abs(gt - prev_gt) > 0.1)
        
        action_idx = action.item()
        agent.set_prev_action(action_idx)
        aux_losses = agent.compute_aux_losses(
            out['h'], agent.prev_h, agent.prev_action_oh,
            out['var_probs'], prev_var_probs, gt, prev_gt,
            action_idx, is_event
        )
        prev_var_probs = out['var_probs'].detach()
        
        transitions.append({
            'obs': obs, 'action': action, 'action_idx': action_idx,
            'log_prob': log_prob, 'entropy': entropy,
            'reward': reward, 'value': out['value'],
            'aux_losses': aux_losses, 'interp': out['interp'],
            'pred_err': out['pred_err'], 'is_event': is_event,
            'action_repeat': agent.action_repeat_count,
            'delta_A_mag': out['delta_A_mag'],
            'pool_size': out['pool_size'],
            'loop_logit': out['loop_logit'],
        })
        
        total_reward += reward
        obs = next_obs
        prev_gt = gt
        gt = next_gt
        if done: break
    
    return transitions, total_reward


def compute_returns(transitions, gamma=0.99):
    returns = []
    R = 0.0
    for t in reversed(transitions):
        R = t['reward'] + gamma * R
        returns.insert(0, R)
    return returns


def update_agent(agent, optimizer, transitions,
                 entropy_coef=0.03, value_coef=0.5, aux_coef=0.3,
                 interp_coef=0.3, loop_bce_coef=1.0):
    returns = compute_returns(transitions)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    if len(returns_t) > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    
    log_probs = torch.stack([t['log_prob'] for t in transitions])
    values = torch.cat([t['value'] for t in transitions]).squeeze(-1)
    entropies = torch.stack([t['entropy'] for t in transitions])
    
    advantages = (returns_t - values.detach()).detach()
    
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns_t)
    entropy_bonus = entropies.mean()
    entropy_pen = torch.clamp(MIN_ENTROPY - entropy_bonus, min=0.0) * 2.0
    
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus + entropy_pen
    
    total_aux = torch.tensor(0.0, device=DEVICE)
    aux_count = 0
    for t in transitions:
        for k, v in t['aux_losses'].items():
            if isinstance(v, torch.Tensor) and v.requires_grad:
                total_aux = total_aux + v
                aux_count += 1
    if aux_count > 0:
        loss = loss + aux_coef * total_aux / max(1, aux_count)
    
    # interp共享头
    shared_targets = []
    shared_outputs = []
    for i in range(1, len(transitions)):
        t = transitions[i]
        pe = t['pred_err']
        r = t['reward']
        conf = 1.0 - min(pe, 1.0)
        expl = min(pe * 3, 1.0)
        explt = min(max(r, 0.0), 1.0)
        shared_targets.append([conf, expl, explt])
        shared_outputs.append(t['interp'][1:])
    if shared_outputs:
        shared_targets_t = torch.tensor(shared_targets, dtype=torch.float32, device=DEVICE)
        shared_outputs_t = torch.stack(shared_outputs)
        loss = loss + interp_coef * F.mse_loss(shared_outputs_t, shared_targets_t)
    
    # 循环头BCE: 高repeat样本加权3x
    loop_logits = []
    loop_targets = []
    loop_weights = []
    for i in range(1, len(transitions)):
        t = transitions[i]
        repeat = t['action_repeat']
        target = min(repeat / 10.0, 1.0)
        loop_logits.append(t['loop_logit'])
        loop_targets.append(target)
        loop_weights.append(1.0 + 2.0 * target)  # repeat高→权重3x
    
    if loop_logits:
        logits_t = torch.stack(loop_logits)
        targets_t = torch.tensor(loop_targets, dtype=torch.float32, device=DEVICE)
        weights_t = torch.tensor(loop_weights, dtype=torch.float32, device=DEVICE)
        per_sample = F.binary_cross_entropy_with_logits(logits_t, targets_t, reduction='none')
        loss = loss + loop_bce_coef * (per_sample * weights_t).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
    optimizer.step()
    
    return {
        'total': loss.item(),
        'policy': policy_loss.item(),
        'value': value_loss.item(),
        'entropy': entropy_bonus.item(),
    }


def train(n_episodes=3000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    envs = [CausalGridWorld(seed=seed+i) for i in range(5)]
    agent = CSMv3Agent().to(DEVICE)  # ← 修GPU bug: 必须to(DEVICE)
    
    optimizer = optim.Adam([
        {'params': agent.encoder.parameters(),         'lr': 3e-4},
        {'params': agent.obj_ssm.parameters(),         'lr': 3e-4},
        {'params': agent.c2_proj.parameters(),         'lr': 3e-4},
        {'params': agent.action_head.parameters(),     'lr': 3e-4},
        {'params': agent.slot_mem.parameters(),        'lr': 3e-4},
        {'params': agent.world_model.parameters(),     'lr': 3e-4},
        {'params': agent.scene_router.parameters(),    'lr': 1e-4},
        {'params': agent.concept_pool.parameters(),    'lr': 1e-4},
        {'params': agent.causal_graph.var_detector.parameters(), 'lr': 1e-4},
        {'params': agent.causal_graph.action_effects,  'lr': 3e-4},
        {'params': agent.causal_graph.var_causal_logits,'lr': 3e-4},
        {'params': agent.meta_cog.low_rank_delta_a.parameters(), 'lr': 3e-4},
        {'params': agent.meta_cog.trace_encoder.parameters(),    'lr': 5e-5},
        {'params': agent.meta_cog.meta_ssm.parameters(),         'lr': 5e-5},
        {'params': agent.meta_cog.loop_scale,                     'lr': 3e-4},  # 单独参数
        {'params': agent.meta_cog.loop_bias,                      'lr': 3e-4},
        {'params': agent.meta_cog.interp_shared.parameters(),    'lr': 1e-4},
        {'params': agent.meta_goal.parameters(),        'lr': 5e-5},
    ], eps=1e-5)
    
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[csm v3.1] 参数量: {n_params:,}")
    print(f"[csm v3.1] 设备: {DEVICE}")
    print(f"[csm v3.1] 修: GPU to(DEVICE) + 单调循环头(abs*scale+bias) + BCE加权")
    
    rewards_log = deque(maxlen=100)
    success_log = deque(maxlen=100)
    history = {'episode': [], 'avg_reward': [], 'success_rate': [],
               'phase': [], 'delta_A_mag': [], 'pool_size': []}
    
    start = time.time()
    metrics = {'policy': 0, 'value': 0, 'entropy': 0}
    
    for ep in range(n_episodes):
        env = envs[ep % len(envs)]
        old_phase = agent.phase
        agent.set_phase(ep)
        if agent.phase != old_phase:
            print(f"  >>> 切换到阶段 {agent.phase} <<<")
        
        traj, total_r = collect_episode(agent, env)
        if len(traj) > 2:
            metrics = update_agent(agent, optimizer, traj)
        
        rewards_log.append(total_r)
        success_log.append(1.0 if total_r > 0.5 else 0.0)
        
        da_mags = [t['delta_A_mag'] for t in traj if t['delta_A_mag'] > 0]
        avg_da = np.mean(da_mags) if da_mags else 0.0
        pool_sizes = [t['pool_size'] for t in traj]
        avg_pool = np.mean(pool_sizes) if pool_sizes else 0
        max_pool = max(pool_sizes) if pool_sizes else 0
        
        if (ep+1) % 100 == 0:
            avg_r = np.mean(rewards_log)
            succ = np.mean(success_log)
            elapsed = time.time() - start
            vram = f"{torch.cuda.memory_allocated()/1024**2:.0f}MB" if torch.cuda.is_available() else "N/A"
            extra = f" | Pool: {avg_pool:.0f}/{max_pool}"
            if agent.phase >= 3: extra += f" | ΔA: {avg_da:.4f}"
            print(f"[csm v3.1] Ep {ep+1}/{n_episodes} P{agent.phase} | "
                  f"AvgR: {avg_r:.3f} | Succ: {succ:.2%} | "
                  f"Ent: {metrics['entropy']:.3f}{extra} | "
                  f"VRAM: {vram} | Time: {elapsed:.1f}s")
            history['episode'].append(ep+1)
            history['avg_reward'].append(float(avg_r))
            history['success_rate'].append(float(succ))
            history['phase'].append(agent.phase)
            history['delta_A_mag'].append(float(avg_da))
            history['pool_size'].append(float(avg_pool))
    
    os.makedirs('results', exist_ok=True)
    with open('results/csm_v3_1_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(agent.state_dict(), 'results/csm_v3_1_model.pt')
    print(f"[csm v3.1] 训练完成, 总耗时: {time.time()-start:.1f}s")
    print(f"[csm v3.1] 最终概念池: {agent.concept_pool_data.shape[0]} 条目")
    return history


if __name__ == '__main__':
    train()
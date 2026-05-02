import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import time, json, os

from env import CausalGridWorld
from csm_agent import CSMv2Agent
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
        out = agent(obs, ext_reward=0.0 if step==0 else transitions[-1]['reward'],
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
            'action_repeat': agent.action_repeat_count,  # 保存实际值
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
                 entropy_coef=0.03, value_coef=0.5, aux_coef=0.3, interp_coef=0.3):
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
    
    # interp 监督: 用连续 action_repeat, 不二值化
    interp_targets = []
    interp_outputs = []
    for i in range(1, len(transitions)):
        t = transitions[i]
        pe = t['pred_err']
        r = t['reward']
        repeat = t['action_repeat']  # 连续值 0-20
        
        # 4 个 target — 连续化!
        loop = min(repeat / 10.0, 1.0)  # 0→1, 跟输入一致
        conf = 1.0 - min(pe, 1.0)
        expl = min(pe * 3, 1.0)
        explt = min(max(r, 0.0), 1.0)  # 正奖励归一化
        
        interp_targets.append([loop, conf, expl, explt])
        interp_outputs.append(t['interp'])
    
    if interp_outputs:
        targets_t = torch.tensor(interp_targets, dtype=torch.float32, device=DEVICE)
        outputs_t = torch.stack(interp_outputs)
        interp_loss = F.mse_loss(outputs_t, targets_t)
        loss = loss + interp_coef * interp_loss
    
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
    agent = CSMv2Agent()
    
    optimizer = optim.Adam([
        {'params': agent.encoder.parameters(), 'lr': 3e-4},
        {'params': agent.obj_ssm.parameters(), 'lr': 3e-4},
        {'params': agent.c2_proj.parameters(), 'lr': 3e-4},
        {'params': agent.action_head.parameters(), 'lr': 3e-4},
        {'params': agent.slot_mem.parameters(), 'lr': 3e-4},
        {'params': agent.world_model.parameters(), 'lr': 3e-4},
        {'params': agent.scene_router.parameters(), 'lr': 1e-4},
        {'params': agent.causal_graph.var_detector.parameters(), 'lr': 1e-4},
        {'params': agent.causal_graph.action_effects, 'lr': 3e-4},
        {'params': agent.causal_graph.var_causal_logits, 'lr': 3e-4},
        {'params': agent.meta_cog.parameters(), 'lr': 5e-5},
        {'params': agent.meta_goal.parameters(), 'lr': 5e-5},
    ], eps=1e-5)
    
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[csm] 参数量: {n_params:,}")
    print(f"[csm] 设备: {DEVICE}")
    print(f"[csm] 修复: interp连续target(不再二值化)")
    
    rewards_log = deque(maxlen=100)
    success_log = deque(maxlen=100)
    history = {'episode': [], 'avg_reward': [], 'success_rate': [], 'phase': []}
    
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
        
        if (ep+1) % 100 == 0:
            avg_r = np.mean(rewards_log)
            succ = np.mean(success_log)
            elapsed = time.time() - start
            vram = f"{torch.cuda.memory_allocated()/1024**2:.0f}MB" if torch.cuda.is_available() else "N/A"
            print(f"[csm] Ep {ep+1}/{n_episodes} P{agent.phase} | "
                  f"AvgR: {avg_r:.3f} | Succ: {succ:.2%} | "
                  f"Ent: {metrics['entropy']:.3f} | "
                  f"VRAM: {vram} | Time: {elapsed:.1f}s")
            history['episode'].append(ep+1)
            history['avg_reward'].append(float(avg_r))
            history['success_rate'].append(float(succ))
            history['phase'].append(agent.phase)
    
    os.makedirs('results', exist_ok=True)
    with open('results/csm_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(agent.state_dict(), 'results/csm_model.pt')
    print(f"[csm] 训练完成, 总耗时: {time.time()-start:.1f}s")
    return history

if __name__ == '__main__':
    train()
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import time, json, os

from env import CausalGridWorld
from csm_agent import CSMv2Agent
from baseline import LSTMBaseline

def collect_episode(agent, env, is_csm=True, max_steps=80):
    """收集一个完整episode"""
    obs = env.reset()
    agent.reset_hidden()
    
    transitions = []
    total_reward = 0
    prev_var_probs = None
    
    for step in range(max_steps):
        out = agent(obs, ext_reward=0.0 if step==0 else transitions[-1]['reward'])
        
        dist = torch.distributions.Categorical(out['action_probs'])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        next_obs, reward, done, _ = env.step(action.item())
        
        # 辅助损失
        aux_losses = {}
        if is_csm:
            agent.set_prev_action(action.item())
            aux_losses = agent.compute_aux_losses(
                out['h'], agent.prev_h if hasattr(agent, 'prev_h') else None,
                agent.prev_action_oh, out['var_probs'], prev_var_probs
            )
            prev_var_probs = out['var_probs'].detach()
        
        transitions.append({
            'obs': obs,
            'action': action,
            'log_prob': log_prob,      # 保留计算图!
            'entropy': entropy,
            'reward': reward,
            'value': out['value'],     # 保留计算图!
            'aux_losses': aux_losses,
        })
        
        total_reward += reward
        obs = next_obs
        if done: break
    
    return transitions, total_reward

def compute_returns(transitions, gamma=0.99):
    """计算折扣回报"""
    returns = []
    R = 0.0
    for t in reversed(transitions):
        R = t['reward'] + gamma * R
        returns.insert(0, R)
    return returns

def update_agent(agent, optimizer, transitions, is_csm=True, 
                 entropy_coef=0.02, value_coef=0.5, aux_coef=0.1):
    """REINFORCE with baseline + 辅助损失"""
    returns = compute_returns(transitions)
    returns_t = torch.tensor(returns, dtype=torch.float32)
    
    # 标准化 returns
    if len(returns_t) > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    
    log_probs = torch.stack([t['log_prob'] for t in transitions])
    values = torch.cat([t['value'] for t in transitions]).squeeze(-1)
    entropies = torch.stack([t['entropy'] for t in transitions])
    
    # === 核心: REINFORCE with baseline ===
    # 策略梯度: ∇J = E[∇log π(a|s) * (R - V(s))]
    # 关键: returns_t 要 detach, values 不 detach (让 value 学习)
    advantages = (returns_t - values.detach()).detach()  # 两者都 detach 给 policy 用
    
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns_t)
    entropy_bonus = entropies.mean()
    
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
    
    # === 辅助损失 ===
    if is_csm:
        total_aux = torch.tensor(0.0)
        for t in transitions:
            for k, v in t['aux_losses'].items():
                if isinstance(v, torch.Tensor):
                    total_aux = total_aux + v
        n_trans = max(1, len(transitions))
        loss = loss + aux_coef * total_aux / n_trans
    
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

def train(agent_type='csm', n_episodes=3000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 多种子环境, 防止过拟合单一布局
    envs = [CausalGridWorld(seed=seed+i) for i in range(5)]
    
    is_csm = agent_type == 'csm'
    agent = CSMv2Agent() if is_csm else LSTMBaseline()
    
    # CSM: 不同模块不同学习率
    if is_csm:
        param_groups = [
            {'params': agent.encoder.parameters(), 'lr': 3e-4},
            {'params': agent.action_head.parameters(), 'lr': 3e-4},
            {'params': agent.obj_ssm.parameters(), 'lr': 3e-4},
            {'params': agent.world_model.parameters(), 'lr': 3e-4},
            {'params': agent.scene_router.parameters(), 'lr': 1e-4},
            {'params': agent.causal_graph.parameters(), 'lr': 1e-4},
            {'params': agent.meta_cog.parameters(), 'lr': 1e-4},  # 慢
            {'params': agent.meta_goal.parameters(), 'lr': 1e-4},
            {'params': agent.slot_mem.parameters(), 'lr': 1e-4},
            {'params': agent.c0_proj.parameters(), 'lr': 3e-4},
            {'params': agent.c1_proj.parameters(), 'lr': 3e-4},
            {'params': agent.c2_proj.parameters(), 'lr': 3e-4},
        ]
        optimizer = optim.Adam(param_groups, eps=1e-5)
    else:
        optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[{agent_type}] 参数量: {n_params:,}")
    
    rewards_log = deque(maxlen=100)
    success_log = deque(maxlen=100)
    history = {'episode': [], 'avg_reward': [], 'success_rate': []}
    
    start = time.time()
    for ep in range(n_episodes):
        env = envs[ep % len(envs)]  # 轮换环境
        traj, total_r = collect_episode(agent, env, is_csm=is_csm)
        
        if len(traj) > 2:  # 至少几步才更新
            metrics = update_agent(agent, optimizer, traj, is_csm=is_csm)
        
        rewards_log.append(total_r)
        success_log.append(1.0 if total_r > 0.5 else 0.0)
        
        if (ep+1) % 100 == 0:
            avg_r = np.mean(rewards_log)
            succ = np.mean(success_log)
            elapsed = time.time() - start
            pol_loss = metrics['policy'] if 'metrics' in dir() else 0
            print(f"[{agent_type}] Ep {ep+1}/{n_episodes} | "
                  f"AvgR: {avg_r:.3f} | Succ: {succ:.2%} | "
                  f"PLoss: {metrics['policy']:.3f} | "
                  f"Ent: {metrics['entropy']:.3f} | "
                  f"Time: {elapsed:.1f}s")
            history['episode'].append(ep+1)
            history['avg_reward'].append(float(avg_r))
            history['success_rate'].append(float(succ))
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{agent_type}_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(agent.state_dict(), f'results/{agent_type}_model.pt')
    print(f"[{agent_type}] 训练完成, 总耗时: {time.time()-start:.1f}s")
    return history

if __name__ == '__main__':
    import sys
    agent_type = sys.argv[1] if len(sys.argv) > 1 else 'csm'
    train(agent_type=agent_type, n_episodes=3000)
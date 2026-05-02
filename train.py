import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import time, json, os

from env import CausalGridWorld
from csm_agent import CSMv2Agent
from baseline import LSTMBaseline

def collect_episode(agent, env, is_csm=True, max_steps=100):
    obs = env.reset()
    agent.reset_hidden()
    trajectory = []
    total_reward = 0
    prev_var_probs = None
    
    for step in range(max_steps):
        out = agent(obs, ext_reward=0.0 if step==0 else trajectory[-1]['reward'])
        action_probs = out['action_probs']
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        next_obs, reward, done, _ = env.step(action.item())
        
        # CSM 加内部奖励
        intrinsic = 0.0
        if is_csm:
            agent.set_prev_action(action.item())
            intrinsic = agent.compute_intrinsic_reward(
                out['curiosity'], out['pred_err'], 
                out['var_probs'], prev_var_probs
            )
            prev_var_probs = out['var_probs'].detach()
        
        trajectory.append({
            'obs': obs, 'action': action.item(), 'reward': reward,
            'log_prob': log_prob, 'value': out['value'],
            'intrinsic': intrinsic,
            'var_probs': out['var_probs'].detach() if is_csm else None,
            'interp': out['interp'].detach() if is_csm else None,
        })
        total_reward += reward
        obs = next_obs
        if done: break
    
    return trajectory, total_reward

def ppo_update(agent, optimizer, trajectory, gamma=0.99, clip=0.2, 
               use_intrinsic=True, entropy_coef=0.01):
    # 计算 returns (包含内部奖励)
    returns = []
    R = 0
    for t in reversed(trajectory):
        r = t['reward']
        if use_intrinsic:
            r += t['intrinsic']
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # PPO loss (简化: 单 epoch)
    log_probs = torch.stack([t['log_prob'] for t in trajectory])
    values = torch.cat([t['value'] for t in trajectory]).squeeze()
    
    advantages = returns - values.detach()
    
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    
    loss = policy_loss + 0.5 * value_loss
    
    # CSM 专属: 因果稀疏损失
    if hasattr(agent, 'causal_graph'):
        loss = loss + 0.001 * agent.causal_graph.sparsity_loss()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
    optimizer.step()
    
    return loss.item()

def train(agent_type='csm', n_episodes=2000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = CausalGridWorld(seed=seed)
    is_csm = agent_type == 'csm'
    agent = CSMv2Agent() if is_csm else LSTMBaseline()
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[{agent_type}] 参数量: {n_params:,}")
    
    rewards_log = deque(maxlen=100)
    success_log = deque(maxlen=100)
    history = {'episode': [], 'avg_reward': [], 'success_rate': []}
    
    start = time.time()
    for ep in range(n_episodes):
        traj, total_r = collect_episode(agent, env, is_csm=is_csm)
        loss = ppo_update(agent, optimizer, traj, use_intrinsic=is_csm)
        
        rewards_log.append(total_r)
        success_log.append(1.0 if total_r > 0.5 else 0.0)
        
        if (ep+1) % 50 == 0:
            avg_r = np.mean(rewards_log)
            succ = np.mean(success_log)
            elapsed = time.time() - start
            print(f"[{agent_type}] Ep {ep+1}/{n_episodes} | "
                  f"AvgR: {avg_r:.3f} | Succ: {succ:.2%} | "
                  f"Loss: {loss:.3f} | Time: {elapsed:.1f}s")
            history['episode'].append(ep+1)
            history['avg_reward'].append(float(avg_r))
            history['success_rate'].append(float(succ))
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    with open(f'results/{agent_type}_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(agent.state_dict(), f'results/{agent_type}_model.pt')
    print(f"[{agent_type}] 训练完成, 总耗时: {time.time()-start:.1f}s")
    return history

if __name__ == '__main__':
    import sys
    agent_type = sys.argv[1] if len(sys.argv) > 1 else 'csm'
    train(agent_type=agent_type, n_episodes=2000)
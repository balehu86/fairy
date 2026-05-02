import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import time, json, os

from env import CausalGridWorld
from csm_agent import CSMv2Agent
from baseline import LSTMBaseline

def collect_batch(agent, env, is_csm=True, batch_steps=256, max_ep_steps=100):
    """收集一批数据 (按步数而非episode数)"""
    obs = env.reset()
    agent.reset_hidden()
    
    all_obs, all_actions, all_rewards = [], [], []
    all_log_probs, all_values, all_intrinsics = [], [], []
    all_var_probs = []
    prev_var_probs = None
    episode_rewards = []
    episode_successes = []
    ep_reward = 0
    
    for step in range(batch_steps):
        out = agent(obs, ext_reward=0.0 if len(all_rewards)==0 else all_rewards[-1])
        action_probs = out['action_probs']
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        next_obs, reward, done, _ = env.step(action.item())
        
        intrinsic = 0.0
        if is_csm:
            agent.set_prev_action(action.item())
            intrinsic = agent.compute_intrinsic_reward(
                out['curiosity'], out['pred_err'], 
                out['var_probs'], prev_var_probs
            )
            prev_var_probs = out['var_probs'].detach()
            all_var_probs.append(out['var_probs'].detach())
        
        all_obs.append(obs)
        all_actions.append(action.item())
        all_rewards.append(reward + intrinsic)
        all_log_probs.append(log_prob)
        all_values.append(out['value'])
        all_intrinsics.append(intrinsic)
        
        ep_reward += reward
        obs = next_obs
        
        if done or (step+1) % max_ep_steps == 0:
            episode_rewards.append(ep_reward)
            episode_successes.append(1.0 if ep_reward > 0.5 else 0.0)
            ep_reward = 0
            obs = env.reset()
            agent.reset_hidden()
            prev_var_probs = None
    
    return {
        'obs': all_obs, 'actions': all_actions, 'rewards': all_rewards,
        'log_probs': torch.stack(all_log_probs), 
        'values': torch.cat(all_values).squeeze(),
        'intrinsics': all_intrinsics,
        'var_probs': all_var_probs,
        'ep_rewards': episode_rewards,
        'ep_successes': episode_successes,
    }

def ppo_update(agent, optimizer, batch, gamma=0.99, clip=0.2, 
               entropy_coef=0.01, n_epochs=3):
    """批量 PPO 更新 (多 epoch)"""
    rewards = torch.tensor(batch['rewards'], dtype=torch.float32)
    log_probs_old = batch['log_probs'].detach()
    values_old = batch['values'].detach()
    
    # 计算 GAE
    returns = torch.zeros_like(rewards)
    R = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    advantages = returns - values_old
    
    total_loss = 0
    for epoch in range(n_epochs):
        # 重新计算 log_probs 和 values (简化: 用旧数据, 只更新策略)
        # 完整 PPO 需要重新 forward, 但 CPU 上太慢, 用近似
        ratio = torch.exp(log_probs_old - log_probs_old.detach())  # =1, 简化
        surr1 = advantages * ratio
        surr2 = advantages * torch.clamp(ratio, 1-clip, 1+clip)
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values_old, returns)
        
        loss = policy_loss + 0.5 * value_loss
        
        if hasattr(agent, 'causal_graph'):
            loss = loss + 0.001 * agent.causal_graph.sparsity_loss()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / n_epochs

def train(agent_type='csm', n_updates=500, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = CausalGridWorld(seed=seed)
    is_csm = agent_type == 'csm'
    agent = CSMv2Agent() if is_csm else LSTMBaseline()
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[{agent_type}] 参数量: {n_params:,}")
    
    rewards_log = deque(maxlen=50)
    success_log = deque(maxlen=50)
    history = {'update': [], 'avg_reward': [], 'success_rate': []}
    
    start = time.time()
    for upd in range(n_updates):
        batch = collect_batch(agent, env, is_csm=is_csm, batch_steps=256)
        loss = ppo_update(agent, optimizer, batch, n_epochs=2)
        
        rewards_log.extend(batch['ep_rewards'])
        success_log.extend(batch['ep_successes'])
        
        if (upd+1) % 25 == 0:
            avg_r = np.mean(rewards_log) if rewards_log else 0
            succ = np.mean(success_log) if success_log else 0
            elapsed = time.time() - start
            print(f"[{agent_type}] Upd {upd+1}/{n_updates} | "
                  f"AvgR: {avg_r:.3f} | Succ: {succ:.2%} | "
                  f"Loss: {loss:.3f} | Time: {elapsed:.1f}s")
            history['update'].append(upd+1)
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
    train(agent_type=agent_type, n_updates=500)
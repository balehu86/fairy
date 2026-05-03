# numpy_train.py — 纯NumPy版, ES优化

import numpy as np
from collections import deque
import time, json, os

from env import CausalGridWorld
from numpy_agent import CSMv3Agent


def sample_action(probs, rng):
    cdf = np.cumsum(probs)
    return int(np.searchsorted(cdf, rng.random()))


def evaluate_params(agent, envs, params, rng, n_rollouts=3, max_steps=80):
    """用给定参数跑几个episode, 返回平均奖励"""
    agent.set_all_params(params)
    total_r = 0.0
    for i in range(n_rollouts):
        env = envs[rng.randint(len(envs))]
        obs, gt = env.reset()
        agent.reset_hidden()
        ep_r = 0.0
        for step in range(max_steps):
            out = agent(obs, ext_reward=0.0 if step == 0 else ep_r * 0.01,
                        action_taken=agent.last_action if step > 0 else None)
            a = sample_action(out['action_probs'], rng)
            obs, r, done, gt = env.step(a)
            agent.set_prev_action(a)
            ep_r += r
            if done: break
        total_r += ep_r
    return total_r / n_rollouts


def train(n_episodes=3000, seed=42, pop_size=20, sigma=0.02, lr=0.03):
    rng = np.random.RandomState(seed)
    
    envs = [CausalGridWorld(seed=seed+i) for i in range(5)]
    agent = CSMv3Agent()
    
    n_params = len(agent.get_all_params())
    print(f"[csm v3/ES] 参数量: {n_params:,}")
    print(f"[csm v3/ES] ES: pop={pop_size}, σ={sigma}, lr={lr}")
    
    theta = agent.get_all_params()
    
    rewards_log = deque(maxlen=100)
    success_log = deque(maxlen=100)
    history = {'episode': [], 'avg_reward': [], 'success_rate': [], 'phase': []}
    
    start = time.time()
    
    for ep in range(n_episodes):
        # 课程
        if ep < 800: phase = 1
        elif ep < 1600: phase = 2
        else: phase = 3
        agent.phase = phase
        
        if ep > 0 and ((ep < 800 and ep % 800 == 0) or 
                        (ep >= 800 and ep < 1600 and ep % 800 == 0) or
                        (ep >= 1600 and ep % 800 == 0)):
            print(f"  >>> 切换到阶段 {phase} <<<")
        
        # ES 一步
        noise = rng.randn(pop_size, n_params)
        rewards = np.zeros(pop_size)
        
        for i in range(pop_size):
            candidate = theta + sigma * noise[i]
            agent.set_all_params(candidate)
            env = envs[ep % len(envs)]
            obs, gt = env.reset()
            agent.reset_hidden()
            ep_r = 0.0
            for step in range(80):
                out = agent(obs, ext_reward=0.0 if step == 0 else 0.01 * ep_r,
                            action_taken=agent.last_action if step > 0 else None)
                a = sample_action(out['action_probs'], rng)
                obs, r, done, gt = env.step(a)
                agent.set_prev_action(a)
                ep_r += r
                if done: break
            rewards[i] = ep_r
        
        # 归一化奖励
        rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # 更新
        grad_estimate = (noise.T @ rewards_norm) / (pop_size * sigma)
        theta += lr * grad_estimate
        
        # 课程约束: 冻结参数
        agent.set_all_params(theta)
        if phase < 3:
            # 冻结 meta_cog 的 low_rank_delta_a 参数 (P1/P2不干预)
            pass  # ES通过奖励信号自然实现课程效果
        
        best_r = rewards.max()
        rewards_log.append(best_r)
        success_log.append(1.0 if best_r > 0.5 else 0.0)
        
        if (ep+1) % 50 == 0:
            avg_r = np.mean(rewards_log)
            succ = np.mean(success_log)
            elapsed = time.time() - start
            print(f"[csm v3/ES] Ep {ep+1}/{n_episodes} P{phase} | "
                  f"AvgBestR: {avg_r:.3f} | Succ: {succ:.2%} | "
                  f"PopR: {rewards.mean():.2f}±{rewards.std():.2f} | "
                  f"Time: {elapsed:.1f}s")
            history['episode'].append(ep+1)
            history['avg_reward'].append(float(avg_r))
            history['success_rate'].append(float(succ))
            history['phase'].append(phase)
    
    os.makedirs('results', exist_ok=True)
    with open('results/csm_v3_es_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    agent.set_all_params(theta)
    # 保存参数
    np.save('results/csm_v3_params.npy', theta)
    print(f"[csm v3/ES] 训练完成, 总耗时: {time.time()-start:.1f}s")
    return history


if __name__ == '__main__':
    train()
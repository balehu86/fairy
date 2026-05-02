import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from env import CausalGridWorld
from csm_agent import CSMv2Agent

def compare():
    """对比两个agent的学习曲线"""
    with open('results/csm_history.json') as f: csm = json.load(f)
    with open('results/baseline_history.json') as f: base = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(csm['episode'], csm['avg_reward'], label='CSM v2', color='purple')
    axes[0].plot(base['episode'], base['avg_reward'], label='PPO+LSTM', color='orange')
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Avg Reward')
    axes[0].set_title('学习曲线'); axes[0].legend(); axes[0].grid(alpha=0.3)
    
    axes[1].plot(csm['episode'], csm['success_rate'], label='CSM v2', color='purple')
    axes[1].plot(base['episode'], base['success_rate'], label='PPO+LSTM', color='orange')
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Success Rate')
    axes[1].set_title('成功率'); axes[1].legend(); axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=100)
    print("对比图保存至 results/comparison.png")

def counterfactual_test():
    """因果推断测试: CSM 能否正确预测干预效果"""
    agent = CSMv2Agent()
    agent.load_state_dict(torch.load('results/csm_model.pt'))
    agent.eval()
    
    env = CausalGridWorld(seed=123)
    obs = env.reset()
    agent.reset_hidden()
    
    # 走几步让状态稳定
    for _ in range(5):
        with torch.no_grad():
            out = agent(obs)
        a = out['action_probs'].argmax().item()
        obs, _, _, _ = env.step(a)
    
    with torch.no_grad():
        out = agent(obs)
        var_probs = out['var_probs']
        
        # 反事实: 如果 has_key=1, door_open 会怎样?
        cf_with_key = agent.causal_graph.counterfactual(var_probs, 0, 1.0)
        cf_without_key = agent.causal_graph.counterfactual(var_probs, 0, 0.0)
    
    print("=== 因果推断测试 ===")
    print(f"当前变量置信度: {var_probs.numpy().round(3)}")
    print(f"干预 has_key=1: door_open={cf_with_key[1]:.3f}")
    print(f"干预 has_key=0: door_open={cf_without_key[1]:.3f}")
    print(f"因果效应 (应 > 0): {(cf_with_key[1] - cf_without_key[1]).item():.3f}")
    
    # 打印因果图
    adj = agent.causal_graph.get_adj().numpy()
    print("\n因果邻接矩阵 (行→列):")
    labels = ['has_key', 'door_open', 'near_treasure', 'saw_decoration']
    print(f"{'':15}" + "".join(f"{l:>15}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:15}" + "".join(f"{adj[i,j]:>15.3f}" for j in range(4)))

def interpretability_demo():
    """展示 S_meta 的可解释信号"""
    agent = CSMv2Agent()
    agent.load_state_dict(torch.load('results/csm_model.pt'))
    agent.eval()
    
    env = CausalGridWorld(seed=999)
    obs = env.reset()
    agent.reset_hidden()
    
    print("=== 元认知可解释信号 (随时间变化) ===")
    print(f"{'Step':>5} {'Action':>8} {'循环警告':>10} {'置信度':>10} {'需要探索':>10} {'需要利用':>10} {'好奇心':>10}")
    
    for step in range(30):
        with torch.no_grad():
            out = agent(obs)
        action = out['action_probs'].argmax().item()
        agent.set_prev_action(action)
        interp = out['interp'].numpy()
        print(f"{step:>5} {action:>8} {interp[0]:>10.3f} {interp[1]:>10.3f} "
              f"{interp[2]:>10.3f} {interp[3]:>10.3f} {out['curiosity']:>10.3f}")
        obs, r, done, _ = env.step(action)
        if done:
            print(f"任务完成于 step {step}, 奖励 {r:.2f}")
            break

if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'compare'
    if cmd == 'compare': compare()
    elif cmd == 'cf': counterfactual_test()
    elif cmd == 'interp': interpretability_demo()
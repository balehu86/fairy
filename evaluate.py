# evaluate.py — v3.1

import torch
import numpy as np
from env import CausalGridWorld
from csm_agent import CSMv3Agent
from device_utils import DEVICE


def counterfactual_test():
    agent = CSMv3Agent()
    agent.load_state_dict(torch.load('results/csm_v3_1_model.pt', weights_only=True, map_location=DEVICE))
    agent.eval()
    
    env = CausalGridWorld(seed=123)
    obs, gt = env.reset()
    agent.reset_hidden()
    
    for _ in range(10):
        with torch.no_grad():
            out = agent(obs, ext_reward=0.0, action_taken=agent.last_action if _ > 0 else None)
        a = out['action_probs'].argmax().item()
        obs, _, done, gt = env.step(a)
        agent.set_prev_action(a)
        if done: break
    
    with torch.no_grad():
        out = agent(obs, ext_reward=0.0)
        var_probs = out['var_probs']
        cf_key1 = agent.causal_graph.counterfactual(var_probs, 0, 1.0)
        cf_key0 = agent.causal_graph.counterfactual(var_probs, 0, 0.0)
    
    print("=== 因果推断测试 ===")
    print(f"变量置信度: {var_probs.cpu().numpy().round(3)}")
    print(f"地面真值:   {gt.round(3)}")
    print(f"do(has_key=1) -> door_open = {cf_key1[1]:.3f}")
    print(f"do(has_key=0) -> door_open = {cf_key0[1]:.3f}")
    print(f"因果效应:   {(cf_key1[1]-cf_key0[1]).item():.3f}")
    
    ae = agent.causal_graph.action_effects.detach().cpu().numpy()
    actions = ['up', 'down', 'left', 'right', 'pickup', 'use']
    labels = ['has_key', 'door_open', 'near_treas', 'saw_deco']
    
    print(f"\n动作直接效果:")
    print(f"{'':>10}" + "".join(f"{l:>12}" for l in labels))
    for i, a in enumerate(actions):
        print(f"{a:>10}" + "".join(f"{ae[i,j]:>12.3f}" for j in range(4)))
    
    adj = agent.causal_graph.get_var_adj().detach().cpu().numpy()
    print(f"\n变量→变量因果:")
    print(f"{'':>14}" + "".join(f"{l:>14}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>14}" + "".join(f"{adj[i,j]:>14.3f}" for j in range(4)))


def interpretability_demo():
    agent = CSMv3Agent()
    agent.load_state_dict(torch.load('results/csm_v3_1_model.pt', weights_only=True, map_location=DEVICE))
    agent.eval()
    
    env = CausalGridWorld(seed=999)
    obs, gt = env.reset()
    agent.reset_hidden()
    
    print("=== 元认知可解释信号 (v3.1 BCE独立头) ===")
    print(f"{'Step':>5} {'Act':>4} {'R':>6} {'Repeat':>7} "
          f"{'循环':>6} {'置信':>6} {'探索':>6} {'利用':>6} {'ΔA':>8} {'Pool':>4}")
    
    for step in range(40):
        with torch.no_grad():
            out = agent(obs, ext_reward=0.0 if step == 0 else 0.0,
                        action_taken=agent.last_action if step > 0 else None)
        action = out['action_probs'].argmax().item()
        agent.set_prev_action(action)
        interp = out['interp'].cpu().numpy()
        da = out['delta_A_mag']
        ps = out['pool_size']
        
        obs, r, done, gt = env.step(action)
        
        print(f"{step:>5} {action:>4} {r:>6.2f} {agent.action_repeat_count:>7.0f} "
              f"{interp[0]:>6.3f} {interp[1]:>6.3f} {interp[2]:>6.3f} {interp[3]:>6.3f} {da:>8.4f} {ps:>4}")
        if done:
            print(f"  >>> 完成 step {step}")
            break


def pool_growth_demo():
    """展示概念池如何随交互增长"""
    agent = CSMv3Agent()
    agent.load_state_dict(torch.load('results/csm_v3_1_model.pt', weights_only=True, map_location=DEVICE))
    agent.eval()
    
    env = CausalGridWorld(seed=42)
    obs, gt = env.reset()
    agent.reset_hidden()
    
    print("=== 概念池增长轨迹 ===")
    print(f"{'Step':>5} {'Act':>4} {'Pool':>4} {'新条目':>6} {'场景':>4}")
    
    prev_size = agent.concept_pool_data.shape[0]
    for step in range(40):
        with torch.no_grad():
            out = agent(obs, ext_reward=0.0, action_taken=agent.last_action if step > 0 else None)
        action = out['action_probs'].argmax().item()
        agent.set_prev_action(action)
        
        new_size = out['pool_size']
        added = "追加" if new_size > prev_size else ""
        prev_size = new_size
        
        obs, r, done, gt = env.step(action)
        
        # 简单场景描述
        has_key = "🔑" if gt[0] > 0.5 else "·"
        door = "🚪" if gt[1] > 0.5 else "·"
        
        print(f"{step:>5} {action:>4} {new_size:>4} {added:>6} {has_key}{door}")
        if done: break


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'cf'
    if cmd == 'cf': counterfactual_test()
    elif cmd == 'interp': interpretability_demo()
    elif cmd == 'pool': pool_growth_demo()
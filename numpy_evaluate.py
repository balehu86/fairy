# numpy_evaluate.py — 纯NumPy版评估

import numpy as np
from env import CausalGridWorld
from numpy_agent import CSMv3Agent


def sample_action_greedy(probs):
    return int(np.argmax(probs))


def counterfactual_test():
    agent = CSMv3Agent()
    theta = np.load('results/csm_v3_params.npy')
    agent.set_all_params(theta)
    agent.phase = 3
    
    env = CausalGridWorld(seed=123)
    obs, gt = env.reset()
    agent.reset_hidden()
    
    for _ in range(10):
        out = agent(obs, ext_reward=0.0, action_taken=agent.last_action if _ > 0 else None)
        a = sample_action_greedy(out['action_probs'])
        obs, _, done, gt = env.step(a)
        agent.set_prev_action(a)
        if done: break
    
    out = agent(obs, ext_reward=0.0)
    var_probs = out['var_probs']
    cf_key1 = agent.causal_graph.counterfactual(var_probs, 0, 1.0)
    cf_key0 = agent.causal_graph.counterfactual(var_probs, 0, 0.0)
    
    print("=== 因果推断测试 ===")
    print(f"变量置信度: {var_probs.round(3)}")
    print(f"地面真值:   {gt.round(3)}")
    print(f"do(has_key=1) -> door_open = {cf_key1[1]:.3f}")
    print(f"do(has_key=0) -> door_open = {cf_key0[1]:.3f}")
    print(f"因果效应:   {cf_key1[1]-cf_key0[1]:.3f}")
    
    ae = agent.causal_graph.action_effects
    actions = ['up', 'down', 'left', 'right', 'pickup', 'use']
    labels = ['has_key', 'door_open', 'near_treas', 'saw_deco']
    
    print(f"\n动作直接效果:")
    print(f"{'':>10}" + "".join(f"{l:>12}" for l in labels))
    for i, a in enumerate(actions):
        print(f"{a:>10}" + "".join(f"{ae[i,j]:>12.3f}" for j in range(4)))
    
    adj = agent.causal_graph.get_var_adj()
    print(f"\n变量→变量因果:")
    print(f"{'':>14}" + "".join(f"{l:>14}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>14}" + "".join(f"{adj[i,j]:>14.3f}" for j in range(4)))


def interpretability_demo():
    agent = CSMv3Agent()
    theta = np.load('results/csm_v3_params.npy')
    agent.set_all_params(theta)
    agent.phase = 3
    
    env = CausalGridWorld(seed=999)
    obs, gt = env.reset()
    agent.reset_hidden()
    
    print("=== 元认知可解释信号 (v3独立头版) ===")
    print(f"{'Step':>5} {'Act':>4} {'R':>6} {'Repeat':>7} "
          f"{'循环':>6} {'置信':>6} {'探索':>6} {'利用':>6} {'ΔA':>6}")
    
    for step in range(40):
        out = agent(obs, ext_reward=0.0 if step == 0 else 0.01 * 0,
                    action_taken=agent.last_action if step > 0 else None)
        action = sample_action_greedy(out['action_probs'])
        agent.set_prev_action(action)
        interp = out['interp']
        da = out['delta_A_mag']
        
        obs, r, done, gt = env.step(action)
        
        print(f"{step:>5} {action:>4} {r:>6.2f} {agent.action_repeat_count:>7.0f} "
              f"{interp[0]:>6.3f} {interp[1]:>6.3f} {interp[2]:>6.3f} {interp[3]:>6.3f} {da:>6.4f}")
        if done:
            print(f"  >>> 完成 step {step}")
            break


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'cf'
    if cmd == 'cf': counterfactual_test()
    elif cmd == 'interp': interpretability_demo()
# CSM v2.1 — 概念状态机架构：完整技术规格与迭代记录

> 本文档是自包含的。发给任何 LLM，它应能理解完整架构、复现代码、继续调试未解决的模块问题、或扩展到新环境。

---

## 一句话总结

CSM 是一个层次化认知架构，用显式状态机替代 Transformer 隐式注意力。经过 7 轮实证迭代，在 7×7 因果网格世界上达到 **69% 成功率**（vs PPO+LSTM baseline 43%），但 69% 主要来自跳跃连接+课程训练+塑形奖励——因果图和元认知模块仍在学习正确表征的路上。

---

## 架构信息流

```
输入 obs (56维)
    │
    ▼
┌──────────┐
│ Encoder  │ 56→64, 2层MLP+GELU
└────┬─────┘
     │ h (64维)──────────────────────┐ (跳跃连接，信息生命线)
     │                               │
     ├──────┬────────────┐           │
     ▼      ▼            ▼           │
┌───────┐ ┌──────┐ ┌──────────┐     │
│ C₋₁   │ │因果图│ │场景路由   │     │
│ 元目标 │ │  G   │ │(加法增强) │     │
│ 8维goal│ │4变量  │ │h+α·enh  │     │
└───┬───┘ │prob  │ └────┬─────┘     │
    │     └──┬───┘      │            │
    │        │          ▼            │
    └────────┤   ┌──────────────┐    │
             │   │  S_obj       │    │
             │   │  GatedSSM    │◀───│── S_meta 调制重置门
             │   │  32维状态     │    │   (delta_signal→r偏置)
             │   └──────┬───────┘    │
             │          │            │
             │          ▼            │
             │   ┌──────────────┐    │
             │   │  槽位记忆     │    │
             │   │  4×64维      │    │
             │   └──────┬───────┘    │
             │          │            │
             ▼          ▼            ▼
        ┌────────────────────────────┐
        │  c2 = Linear(S_obj,        │
        │      slot_read, h)         │
        │  32+64+64 → 64            │
        └────────────┬───────────────┘
                     ▼
              ┌─────────────┐
              │  ActionHead  │
              │  π(a|c2), V  │
              └─────────────┘

隐藏的第二个流：

S_obj ──delta──▶ S_meta (GatedSSM, 16维)
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     delta_signal   interp(4维)   目标更新
     (SSM路径,      (前馈路径,    信号
      调制S_obj      不走SSM)
      重置门)
```

---

## 各模块完整规格

### 1. Encoder (56→64)

```python
self.net = nn.Sequential(
    nn.Linear(obs_dim, hidden),   # 56→64
    nn.GELU(),
    nn.Linear(hidden, hidden),    # 64→64
)
```

输入：3×3 部分可观测视野 one-hot (6类×9格=54维) + has_key + door_open (2维) = 56维。

### 2. C₋₁ 元目标层 (8维)

```python
signals = torch.tensor([curiosity, capability_gap, ext_reward])  # 3维
goal = tanh(Linear(concat(h.detach(), signals)))  # 67→8
```

三个控制信号：
- **好奇心**：学习进度 `max(0, older_avg_err − recent_avg_err)`，取最近 20 步分前后 10 步。不用原始预测误差，防噪声电视效应
- **能力缺口**：近期负奖励统计
- **外部奖励**：环境即时奖励

课程控制：P1/P2 阶段 `goal *= 0`（冻结），P3 解锁。初始化时不干扰下游。

### 3. 因果图 G

#### 3.1 变量检测器

4 个离散变量：has_key, door_open, near_treasure, saw_deco

```python
var_probs = sigmoid(Linear(h))  # 64→4
```

用环境地面真值 `BCE(var_probs, gt)` 监督。验证结果：v_key≈0.833, v_door≈0.620。**必须用地面真值监督**——纯自监督下检测器不知道该检测什么。

#### 3.2 转移预测

```python
direct = action_oh @ action_effects     # 动作直接效果 (6×4 参数矩阵)
mediated = direct @ var_adj             # 一步因果传播 (4×4 sigmoid矩阵)
delta = tanh(direct + mediated)
```

关键设计决策及理由：

| 决策 | 理由 | 之前失败的方式 |
|------|------|--------------|
| 预测增量而非绝对值 | 预测绝对值时偏置≈target即可解，A矩阵无梯度 | 全 0.47 均匀 |
| action_effects 直接参数化 | sigmoid(logit)≈0.5 均匀化，直接参数从 0 开始只有正梯度才能推离 | 全 0.47 均匀 |
| 事件步训练 (10x权重) | 99.98%的步 delta=0，训练它们鼓励模型输出 0 淹没真信号 | action_effects 全零 |
| sparsity 只作用于 var_adj | action_effects 如果也加稀疏约束，少量事件步的正梯度打不过每步的稀疏推力 | action_effects 全零 |

**当前状态：action_effects 在最新一轮仍全≈0，event_supervision_loss 的效果待验证。**

#### 3.3 反事实推理

```python
# do(X=x): 固定干预变量，切断入边，传播因果效应
modified[intervene_idx] = intervene_val
adj[:, intervene_idx] = 0
delta = modified - var_probs
propagated = delta @ adj
result = clamp(var_probs + propagated, 0, 1)
```

### 4. 场景感知稀疏路由

**v2.0 乘法掩码（已废弃）**：
```python
h_out = h * sparse_mask  # 丢弃 85% 维度，信息无法恢复
```

**v2.1 加法增强（当前版本）**：
```python
enhancement = Σ(weights_i × enhancement_vector_i)  # 16个场景原型的加权组合
alpha = sigmoid(gate(concat(h, goal)))              # 门控强度
h_out = h + alpha × enhancement                     # 残差增强
```

关键性质：**初始化时退化为恒等映射**。enhancement_vectors 初始化为 0，所以 h_out ≈ h。学习是渐进增强，不是先破坏再重建。

这是从 6%→69% 的最大单项改进。

### 5. S_obj GatedSSM (state_dim=32)

**v2.0 裸递归（已废弃）**：`S' = A·S + B·x`，梯度爆炸/消失，Loss 剧烈震荡。

**v2.1 门控结构（当前版本）**：
```python
z = sigmoid(W_z · concat(x, S))         # 遗忘门
r = sigmoid(W_r · concat(x, S) + Dr)    # 重置门（受 S_meta 调制）
S_new = (1−z)·S + z·tanh(W_h · concat(x, r·S))
y = output_proj(S_new)
```

Dr 由 S_meta 的 delta_signal 提供，作为**重置门偏置**。这比直接修改 A 矩阵稳定——偏置只影响"忘多少"，不改变系统动力学基本结构。

### 6. S_meta 元认知层 (state_dim=16)

核心原则：S_meta **不处理外部输入**，只观察和干预**推理过程**。

**v2.0 的问题**：输入 S_obj 绝对值，步间几乎恒定 → SSM 进入不动点 → interp 全锁 ~0.5。

**v2.1 的修复**：

输入：`delta_S_obj` (32维) + `action_repeat_count` (1维, 归一化到 0-2) + `pred_err, confidence, entropy` (3维) = 36维

delta_S_obj 在状态变化时非零；循环时归零但 action_repeat_count 递增——两个信号互补。

**双路输出**：

1. **delta_signal → SSM 路径**：需要时间平滑，走 GatedSSM，调制 S_obj 的重置门
2. **interp → 前馈路径**：
```python
interp_input = tensor([action_repeat, pred_err, confidence, entropy])  # 只用4个有效信号!
interp = Sequential(Linear(4,16), ReLU, Linear(16,4), Sigmoid)(interp_input)
```

为什么 interp 不走 SSM：SSM 的动力学会让恒定输入的输出收敛到不动点。前馈路径保证输入变化→输出立即变化。

为什么只用 4 个输入而非 36 个：delta_S_obj 大部分时间全零（32/36 维噪声），会淹没 4 个有效信号。

interp 4 维信号及监督：

| 通道 | 含义 | 监督 target |
|------|------|------------|
| interp[0] | 循环警告 | 当前动作 == 上一步动作? |
| interp[1] | 置信度 | 1 − 预测误差 |
| interp[2] | 需要探索 | min(预测误差×3, 1) |
| interp[3] | 需要利用 | 正奖励? |

**辅助监督**：`MSE(interp, observable_targets) × 0.2`。自指系统仍需要知觉锚点——无监督 interp 会学出不动点。

**当前状态：interp 前馈版本解决了不动点问题（循环从 0.27 缓慢降到 0.27，置信度从 0.57 升到 0.60），但响应幅度太小，没有随 action_repeat 强烈递增。需要更强的监督权重或架构调整。**

### 7. 槽位工作记忆 (4 槽位, 64维)

```python
# 读: 注意力
attn = softmax(slots @ query / √d)    # (4,)
read = Σ(attn_i × slot_i)

# 写: 软门控
write_weights = softmax(Linear(query))  # (4,)
slots += write_weights × (query − slots) × 0.3  # 慢更新
```

初始化为可学习的 empty_token。在 80 步短 episode 中难单独评估，长序列任务中应发挥关键作用。

### 8. 跳跃连接

```python
c2 = Linear(concat(S_obj(32), slot_read(64), h(64)))  # 160→64
```

**69% 成功率的最大功臣**。即使 SSM、槽位、场景路由全失败，信息通过 h 直通到 ActionHead。ResNet 洞见：**深层模块应该是残差/增强，不是替代**。

### 9. 内部奖励

```python
R_internal = 0.5 × learning_progress + 0.3 × info_gain
R_internal = min(R_internal, 0.5)  # 封顶防失控
```

- 学习进度：`max(0, 旧10步平均误差 − 新10步平均误差)`
- 信息增益：因果变量置信度的平均绝对变化

### 10. 控制符系统

时间节律/导航/结构/生成。当前简化版本未显式实现，已预留接口。

---

## 训练范式

### 课程三阶段

| 阶段 | Episodes | 活跃 | 冻结 |
|------|----------|------|------|
| P1 核心 | 0–799 | Encoder, SSM, ActionHead, WorldModel, Slot, 因果检测器 | S_meta 干预、目标调制、因果转移 |
| P2 因果 | 800–1599 | +因果转移预测、因果稀疏 | S_meta 干预 |
| P3 元认知 | 1600+ | 全部解锁 | 无 |

课程的理由：**鸡生蛋问题**。S_meta 需要 S_obj 已会做事才能学到元认知，但 S_meta 的干扰又让 S_obj 学不会。因果图需要检测器先学会变量才能训转移。课程让每个模块在前置就绪后再启动。

### 损失函数

```
L = L_policy + 0.5×L_value + 0.03×H(π) + entropy_floor_penalty
  + 0.3 × (L_world + L_causal_detect [+P2: L_event_supervision + L_causal_sparse])
  + 0.2 × L_interp
```

各损失：

| 损失 | 公式 | 适用阶段 |
|------|------|---------|
| L_policy | REINFORCE: −Σ(log π(a) × (R−V).detach()) | P1+ |
| L_value | MSE(V, normalized_returns) | P1+ |
| H(π) | 策略熵奖励 | P1+ |
| entropy_floor | max(0, 0.3−H) × 2.0 | P1+ |
| L_world | MSE(world_model(h,a), h_next) | P1+ |
| L_causal_detect | BCE(var_probs, ground_truth) | P1+ |
| L_event_supervision | 因果图.event_supervision_loss(action_idx, gt_delta) × 5.0 | P2+, 事件步 |
| L_causal_sparse | L0.5(var_adj) × 0.2 | P2+ |
| L_interp | MSE(interp, [loop,conf,expl,explt]) | P1+ |

### 优化器配置

```python
optimizer = Adam([
    {'params': encoder.parameters(),         'lr': 3e-4},
    {'params': obj_ssm.parameters(),         'lr': 3e-4},
    {'params': c2_proj.parameters(),         'lr': 3e-4},
    {'params': action_head.parameters(),     'lr': 3e-4},
    {'params': slot_mem.parameters(),        'lr': 3e-4},
    {'params': world_model.parameters(),     'lr': 3e-4},
    {'params': scene_router.parameters(),    'lr': 1e-4},
    {'params': causal_graph.var_detector,    'lr': 1e-4},
    {'params': causal_graph.action_effects,  'lr': 3e-4},  # 高学习率
    {'params': causal_graph.var_causal_logits,'lr': 1e-4},
    {'params': meta_cog.parameters(),        'lr': 5e-5},
    {'params': meta_goal.parameters(),       'lr': 5e-5},
], eps=1e-5)

grad_clip = 0.5  # max_norm
```

### 训练配置

- 5 个不同种子的 CausalGridWorld 轮换，防过拟合
- GPU (CUDA) 和 CPU 均支持
- 典型耗时：~300s(GPU) / ~1200s(CPU)

---

## 环境：7×7 因果网格世界

```
┌───────────────┐
│ . . . │ . . T │
│ . K . │ . . . │  K=钥匙  D=门  T=宝藏  5=装饰(干扰)
│ . . . D . . . │
│ . . . │ . . . │
│ . A . │ . . 5 │  A=Agent
└───────────────┘
```

因果链：`has_key=True → can_open_door=True → can_reach_treasure=True`

- **部分可观测**：3×3 视野 (6类 one-hot × 9格) + has_key + door_open = 56维
- **6 动作**：up/down/left/right/pickup/use
- **塑形奖励**：靠近子目标 +0.1/步，捡钥匙 +0.5，开门 +0.5，宝藏 +1.0
- **步长惩罚**：-0.01/步
- **干扰变量**：随机装饰，与任务无因果但可能存在统计相关
- **地面真值**：`env.get_ground_truth()` → [has_key, door_open, near_treasure, saw_deco]
- **干预接口**：`env.intervene(variable, value)` 用于反事实测试

---

## 完整迭代记录：7 轮实证

每一轮的问题、症状、诊断和修复。**这是防止重复犯错的关键信息。**

### 第 1 轮：裸架构崩溃

| 项 | 值 |
|----|-----|
| 结果 | CSM 12%, Baseline 25% |
| 症状 | Loss 震荡 -0.3 到 1.5, 成功率锁 12% |
| 病根 | 裸递归 SSM `S'=A·S+B·x` 特征值不稳定；单 episode PPO 方差巨大 |
| 修复 | GRU 门控替代裸递归；批量 PPO；梯度裁剪 |

### 第 2 轮：策略无梯度

| 项 | 值 |
|----|-----|
| 结果 | Loss 锁死 0.42, 成功率 6% |
| 症状 | policy_loss 完全消失，只有 value_loss 在更新 |
| 病根 | `ratio = exp(log_old - log_old.detach())` 永远等于 1，策略参数零梯度 |
| 修复 | 改用 REINFORCE with baseline |

### 第 3 轮：奖励太稀疏

| 项 | 值 |
|----|-----|
| 结果 | CSM 12%, Baseline 25% |
| 症状 | 94% episode 无正信号，agent 学不到因果链 |
| 病根 | 只有宝藏 +1.0 太稀疏；无子目标塑形 |
| 修复 | 加塑形奖励（靠近子目标 +0.1/步）+ 子目标里程碑（钥匙 +0.5, 门 +0.5） |

### 第 4 轮：信息瓶颈

| 项 | 值 |
|----|-----|
| 结果 | CSM 29%, Baseline 43% |
| 症状 | 成功率爬升但始终落后 baseline |
| 病根 | 场景路由的乘法掩码丢弃 85% 信息，下游无信号；模块串联=串联瓶颈 |
| 修复 | 乘法掩码→加法增强；加跳跃连接 `c2 = Linear(S_obj, slot, h)` |

**加跳跃连接后第 4 轮结果**：CSM 69%, Baseline 43%。这是决定性突破。

### 第 5 轮：因果图和元认知不动点

| 项 | 值 |
|----|-----|
| 结果 | CSM 69%, 因果图全 0.47, interp 全锁 0.5 |
| 症状 | adj 矩阵均匀如随机；interp 不随输入变化 |
| 病根 | 预测绝对值→偏置即可解；S_obj 绝对值输入→SSM 不动点 |
| 修复 | 预测增量 delta；输入改 delta_S_obj；课程训练；因果检测器加地面真值监督；interp 加 MSE 辅助监督 |

### 第 6 轮：因果图仍然均匀

| 项 | 值 |
|----|-----|
| 结果 | adj 仍全 0.47, 因果效应 0.025 |
| 症状 | predict_delta 的间接梯度不够强 |
| 病根 | sigmoid(logit)≈0.5 是梯度甜蜜点，稀疏损失推小 vs 预测损失推大→平衡在 0.47；事件步训练后仍被非事件步淹没 |
| 修复 | action_effects 直接参数化(不走sigmoid，从0开始)；事件步直接监督(event_supervision_loss)；sparsity 只作用于 var_adj |

### 第 7 轮：interp 衰减而非递增 + action_effects 全零

| 项 | 值 |
|----|-----|
| 结果 | interp 循环 0.37→0.27(递减!); action_effects 全≈0; 因果效应 0.429(假阳性) |
| 症状 | interp 走前馈后仍有问题——4个输入中 action_repeat 权重不够 |
| 病根 | interp 输入从 36 维降到 4 维后，delta_S_obj 不再作为输入但 interp 仍需感知"发生了什么"；action_effects 的 event_supervision 可能权重不够 |
| 当前状态 | **未解决** |

---

## 未解决的问题（优先级排序）

### P0: 因果图 action_effects 仍为零

**现象**：`action_effects` 矩阵全≈0，即使有 event_supervision_loss

**可能原因**：
1. event_supervision 损失权重 5.0 不够——事件步太少(~3步/episode × ~50步)，5.0 × 3/50 = 0.3 等效权重，低于 sparsity 的推力
2. 稀疏损失的 var_adj 部分(0.2)虽然不直接作用于 action_effects，但 var_adj≈0.5 时 mediated 路径 `direct @ adj` 的梯度会被稀释
3. action_effects 的学习率 3e-4 可能需要更高

**建议尝试**：
- 将 event_supervision 权重从 5.0 提到 50.0
- 完全断开 mediated 路径，只训练 direct 路径，等 action_effects 学好再加回来
- 用硬编码的 action_effects 做冷启动：`action_effects[4,0] = 0.5` (pickup→has_key), `action_effects[5,1] = 0.5` (use→door_open)

### P1: interp 对 action_repeat 响应不够

**现象**：重复 20 次同一动作时，循环警告从 0.27 降到 0.27（应该升到 >0.5）

**可能原因**：
1. interp 监督 target 的"循环"定义为 `prev_action == curr_action`，但序列中大部分时候都是 1.0，网络学到输出恒定值
2. 更好的 target 应该是 `min(1.0, repeat_count / 5.0)`，与 action_repeat 强度正相关

**建议尝试**：
- 改变 interp 监督：`loop_target = min(1.0, action_repeat / 5.0)`
- 增大 interp_coef 从 0.2 到 1.0
- 给 interp_ff 加一层隐藏层：`Linear(4,16,ReLU,Linear(16,16),ReLU,Linear(16,4),Sigmoid)`

### P2: 元认知 delta_signal 对策略无可见影响

**现象**：P3 解锁后 delta_signal 开始调制 S_obj 重置门，但成功率无明显额外提升

**可能原因**：delta_signal 的幅度太小（×0.1 的偏置调制），对 SSM 行为几乎无影响

**建议尝试**：
- 将 delta_signal 的调制系数从 0.1 提到 0.5
- 给 delta_signal 加辅助监督：当 interp 检测到循环时，delta_signal 应该输出"增大重置"的信号

---

## 关键代码模式

### 必须遵循的模式

```python
# 1. 加法增强（不是乘法掩码！）
h_out = h + alpha * enhancement  # 不是 h * mask

# 2. 跳跃连接（h 直通到最终投影）
c2 = Linear(concat(S_obj, slot_read, h))  # h 必须在！

# 3. 门控 SSM（不是裸递归！）
S_new = (1-z)*S + z*tanh(W_h·concat(x, r*S))  # 不是 S' = A·S + B·x

# 4. 预测增量（不是绝对值！）
delta = predict_delta(prev_vars, action)  # 不是 predict(current_vars)

# 5. 直接参数化因果效果（不走 sigmoid！）
action_effects = Parameter(zeros(n_actions, n_vars))  # 不是 sigmoid(logit)

# 6. interp 走前馈（不走 SSM！）
interp = FF_network(scalars_only)  # 不是 SSM_output

# 7. 事件步训练因果转移（不是每步训练！）
if is_event_step: train_causal_transfer()  # 10x 权重

# 8. 地面真值监督检测器
BCE(var_probs, ground_truth)  # 不是无监督

# 9. 学习进度型好奇心
curiosity = max(0, older_err - recent_err)  # 不是 raw prediction error

# 10. 熵地板
if entropy < 0.3: penalty += 2.0  # 防策略坍缩

# 11. 课程训练
P1: 只训核心; P2: 加因果; P3: 加元认知  # 不是一开始全开

# 12. GPU device 一致性
torch.tensor([...], device=tensor.device)  # 所有手动创建的 tensor
```

### 绝对不能重复的错误

```python
# ❌ 乘法掩码
h_out = h * sparse_mask  # 丢弃信息，不可恢复

# ❌ 裸递归
S' = A @ S + B(x)  # 梯度爆炸/消失

# ❌ PPO ratio=1 假更新
ratio = exp(old_logp - old_logp.detach())  # 永远=1

# ❌ 预测绝对值
predict(var_probs) → var_probs  # 偏置≈target 即可解

# ❌ sigmoid 参数化因果边
sigmoid(logit)  # 平衡在 0.47, 学不出稀疏结构

# ❌ interp 走 SSM
SSM(huge_dim_input) → interp  # 恒定输入→不动点

# ❌ 每步训练因果转移
if not event: loss = MSE(0, 0)  # 鼓励模型输出 0

# ❌ S_meta 输入 S_obj 绝对值
S_meta_input = S_obj  # 步间不变→不动点

# ❌ sparsity 同时作用于 action_effects
L1(action_effects)  # 事件步正梯度打不过每步稀疏推力

# ❌ 奖励只用稀疏终点
reward = 1.0 if treasure else 0  # 94% 无信号

# ❌ 无地面真值监督因果检测器
var_probs = self_supervised(h)  # 不知道该检测什么
```

---

## 文件结构

```
csm_v2/
├── device_utils.py      # GPU/CPU 自动检测
├── env.py               # 因果网格世界（含地面真值、干预接口）
├── modules.py           # 所有神经网络模块定义
├── csm_agent.py         # CSM v2.1 完整智能体
├── train.py             # 课程训练循环
└── evaluate.py          # 因果推断 + 元认知可视化
```

---

## 运行

```bash
pip install torch numpy matplotlib

# 训练 (~300s GPU / ~1200s CPU)
python train.py

# 因果推断测试（检查 action_effects 和 var_adj）
python evaluate.py cf

# 元认知可解释性（检查 interp 随 repeat 的响应）
python evaluate.py interp
```

---

## 模块当前状态

| 模块 | 功能 | 验证数据 | 备注 |
|------|------|---------|------|
| Encoder + SSM + ActionHead | ✅ 正常 | 69% 成功率 | 核心路径可靠 |
| 跳跃连接 | ✅ 正常 | 移除后暴跌 | 信息生命线 |
| 因果图检测器 | ✅ 正常 | v_key≈0.83, v_door≈0.62 | 地面真值监督有效 |
| 因果图转移 | ❌ 未解决 | action_effects 全≈0 | 直接参数化+事件步监督，效果不足 |
| 场景路由 | ✅ 正常 | 加法增强 | 初始化为恒等映射 |
| 元认知 interp | ⚠ 部分 | 循环信号有变化但幅度小 | 前馈路径解决不动点，响应不够强 |
| 元认知 delta_signal | ⚠ 待验证 | 调制 S_obj 重置门 | P3 解锁后影响尚需量化 |
| 槽位记忆 | ⚠ 存在 | 短 episode 难评估 | 长序列任务中应发挥作用 |
| 内部奖励 | ✅ 正常 | 学习进度型 | 封顶 0.5 防失控 |

---

## 下一步

1. **修复因果图**：尝试硬编码冷启动 action_effects[4,0]=0.5, action_effects[5,1]=0.5，然后让微调接管
2. **修复 interp**：改变监督 target 为 `min(1.0, repeat/5.0)`，增大权重
3. **多房间环境**：CSM 的架构优势在长序列复杂任务中才会真正显现
4. **量化元认知调制**：测量 P3 解锁后 delta_signal 对 S_obj 行为的实际影响
5. **睡眠/巩固机制**：推理时记录高奖励轨迹，周期性回放更新参数
6. **时间对齐层**：多模态输入时间尺度差异需要对齐

---

## 诚实的结论

69% vs 43% 证明了**架构方向正确**，但还没有证明每个模块都在独立贡献。跳跃连接做了大部分重活，高级模块更像是"没拖后腿"而非"在帮忙"。

但这是所有认知架构的必经之路——先让系统不崩溃，再让每个模块学到正确表征，最后让它们协同产生涌现行为。我们现在在第二步。

真正的分水岭在多房间环境：当 episode 从 80 步变成 500 步，当需要跨房间记住钥匙位置，当需要在多个子目标间切换——那时工作记忆、因果推理、元认知才会从"没拖后腿"变成"不可或缺"。

---

## 附录：可验证的数字

| 指标 | CSM v2.1 | PPO+LSTM Baseline |
|------|---------|------------------|
| 参数量 | 47,108 | 89,959 |
| 最终成功率 | 69% | 43% |
| 最终平均奖励 | 0.967 | 0.362 |
| P1 结束时成功率 | 26% | ~25% |
| P2 结束时成功率 | 38% | ~35% |
| P3 峰值成功率 | 69% | ~50% |
| 训练耗时(CPU) | ~1100s | ~300s |
| 因果图 var_adj | 全≈0.47 | N/A |
| action_effects | 全≈0 | N/A |
| 因果效应 do(key→door) | 0.429(假阳性) | N/A |
| interp 循环响应 | 0.27→0.27(几乎无) | N/A |
| var_key 检测精度 | 0.833 | N/A |
| var_door 检测精度 | 0.620 | N/A |
# CSM v2.2 — 概念状态机架构：完整技术规格与迭代记录

> 本文档是自包含的。发给任何 LLM，它应能理解完整架构、复现代码、继续调试未解决的模块问题、或扩展到新环境。所有数字来自实证，不是理论推测。

---

## 一句话总结

CSM 是一个层次化认知架构，用显式状态机替代 Transformer 隐式注意力。经过 8 轮实证迭代，在 7×7 因果网格世界上达到 **74% 成功率**（vs PPO+LSTM baseline 43%）。因果图的动作效果矩阵首次学到了正确表征（pickup→has_key=0.249, use→door_open=0.278），但变量→变量因果邻接仍均匀（~0.45），interp 响应幅度不足。高级模块正在从"不拖后腿"向"独立贡献"过渡。

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
             │   │  32维状态     │    │   (delta_signal→r偏置×0.1)
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
      调制S_obj      不走SSM,
      重置门)        3输入→16→16→4)

因果图内部结构：

action_oh ──▶ action_effects(6×4) ──▶ direct ──┐
                                               │  tanh(direct+mediated)
var_probs ──▶ var_adj(4×4 sigmoid) ──▶ mediated=direct@adj ──┘

地面真值监督流：
env.step() ──▶ gt, next_gt ──▶ BCE(detect(h), gt)     # 检测器
                            ──▶ event_supervision(a, gt_delta)  # 动作→变量
                            ──▶ var_causal_supervision(gt_delta) # 变量→变量
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

用环境地面真值 `BCE(var_probs, gt)` 监督。验证结果：v_key≈0.83, v_door≈0.62。**必须用地面真值监督**——纯自监督下检测器不知道该检测什么。

#### 3.2 动作→变量效果 (action_effects)

```python
action_effects = Parameter(zeros(n_actions, n_vars))  # 6×4, 直接参数化
direct = action_oh @ action_effects
```

**不走 sigmoid**。从 0 开始，只有正梯度才能推离 0。

第 8 轮验证结果：
```
        has_key  door_open  near_treas  saw_deco
pickup    0.249     0.148      0.053    -0.009  ← has_key 最大!
use       0.208     0.278      0.082     0.017  ← door_open 最大!
```

训练方式：`event_supervision_loss`，只在因果变量变化的"事件步"训练。损失公式：

```python
changed = (gt_delta.abs() > 0.05).float()  # 哪些变量变了
target = gt_delta * changed                 # 只监督变化变量
mask = changed + 0.1                        # 未变化变量给小权重
loss = MSE(action_oh @ action_effects, target) * mask
```

**不加 sparsity**。之前 sparsity 同时作用于 action_effects 时，每步的推力(~50步)远大于事件步的正梯度(~3步)，结果 action_effects 全被推向 0。

#### 3.3 变量→变量因果 (var_adj)

```python
var_causal_logits = Parameter(zeros(n_vars, n_vars))  # 初始化打破对称
var_adj = sigmoid(var_causal_logits) * (1 - eye)       # 无自环
```

**先验初始化**（v2.3 新增）：
```python
with torch.no_grad():
    var_causal_logits[0, 1] = 2.0   # has_key→door_open 先验大, sigmoid(2)≈0.88
    var_causal_logits[3, :] = -2.0  # saw_deco→其他 先验小, sigmoid(-2)≈0.12
```

**共现监督**（v2.3 新增）：
```python
def var_causal_supervision_loss(gt_deltas):
    changed = (gt_deltas.abs() > 0.05).float()
    co_occurred = changed.unsqueeze(0) * changed.unsqueeze(1)  # 两个变量同时变化=1
    target = co_occurred * (1 - eye)
    loss = BCE(var_adj, target.detach())  # 给每个边独立的梯度方向
```

逻辑：如果变量 i 和 j 同时变化，i→j 这条边应该存在。共现≠因果，但在简单环境中方向基本正确。

**当前状态：var_adj 仍全 ~0.45，共现监督+先验的效果待下一轮验证。**

#### 3.4 转移预测

```python
mediated = direct @ var_adj              # 一步因果传播
delta = tanh(direct + mediated)
```

#### 3.5 反事实推理

```python
modified[intervene_idx] = intervene_val  # 强制干预
adj[:, intervene_idx] = 0                # 切断入边
delta = modified - var_probs
propagated = delta @ adj
result = clamp(var_probs + propagated, 0, 1)
```

**注意**：当 var_adj 全 ~0.45 时，任何干预都会等量传播到所有变量，产生假阳性因果效应。只有 var_adj 变得稀疏后，反事实推理才有意义。

### 4. 场景感知稀疏路由

**v2.0 乘法掩码（已废弃）**：
```python
h_out = h * sparse_mask  # 丢弃 85% 维度，信息无法恢复
```

**v2.1+ 加法增强（当前版本）**：
```python
enhancement = Σ(weights_i × enhancement_vector_i)  # 16个场景原型的加权组合
alpha = sigmoid(gate(concat(h, goal)))              # 门控强度
h_out = h + alpha × enhancement                     # 残差增强
```

关键性质：**初始化时退化为恒比映射**。enhancement_vectors 初始化为 0，所以 h_out ≈ h。学习是渐进增强，不是先破坏再重建。

这是从 6%→69% 的最大单项改进。

### 5. S_obj GatedSSM (state_dim=32)

**v2.0 裸递归（已废弃）**：`S' = A·S + B·x`，梯度爆炸/消失，Loss 剧烈震荡。

**v2.1+ 门控结构（当前版本）**：
```python
z = sigmoid(W_z · concat(x, S))         # 遗忘门
r = sigmoid(W_r · concat(x, S) + Dr)    # 重置门（受 S_meta 调制）
S_new = (1−z)·S + z·tanh(W_h · concat(x, r·S))
y = output_proj(S_new)
```

Dr 由 S_meta 的 delta_signal 提供，作为**重置门偏置**。这比直接修改 A 矩阵稳定——偏置只影响"忘多少"，不改变系统动力学基本结构。

### 6. S_meta 元认知层 (state_dim=16)

核心原则：S_meta **不处理外部输入**，只观察和干预**推理过程**。

**输入**：`delta_S_obj` (32维) + `action_repeat_count` (1维, 归一化到 0-2) + `pred_err, confidence, entropy` (3维) = 36维

delta_S_obj 在状态变化时非零；循环时归零但 action_repeat_count 递增——两个信号互补。

**双路输出**：

1. **delta_signal → SSM 路径**：需要时间平滑，走 GatedSSM，调制 S_obj 的重置门
2. **interp → 前馈路径**（v2.3 当前版本）：
```python
# 只用 3 个强信号! 不用 delta_S_obj (大部分时间全零=噪声)
interp_input = tensor([action_repeat, pred_err, entropy])  # 3维
interp = Sequential(
    Linear(3, 16), ReLU,
    Linear(16, 16), ReLU,
    Linear(16, 4), Sigmoid
)(interp_input)
```

为什么 interp 不走 SSM：SSM 的动力学会让恒定输入的输出收敛到不动点。前馈路径保证输入变化→输出立即变化。

为什么只用 3 个输入：delta_S_obj 大部分时间全零（32/36 维噪声），confidence 与 pred_err 线性相关（冗余），会淹没有效信号。

interp 4 维信号及监督：

| 通道 | 含义 | 监督 target |
|------|------|------------|
| interp[0] | 循环警告 | `min(1.0, action_repeat / 5.0)`（与 repeat 强度正相关） |
| interp[1] | 置信度 | 1 − 预测误差 |
| interp[2] | 需要探索 | min(预测误差×3, 1) |
| interp[3] | 需要利用 | 正奖励? |

**当前状态：interp 有响应但不强（循环 0.45→0.43 随 repeat 20步）。监督 target 的"循环"定义需要从"是否相同动作"改为"repeat 计数的标准化值"。**

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

**69%→74% 成功率的关键基础设施**。即使 SSM、槽位、场景路由全失败，信息通过 h 直通到 ActionHead。ResNet 洞见：**深层模块应该是残差/增强，不是替代**。

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
| P2 因果 | 800–1599 | +因果转移预测、因果稀疏、var_causal共现 | S_meta 干预 |
| P3 元认知 | 1600+ | 全部解锁 | 无 |

课程的理由：**鸡生蛋问题**。S_meta 需要 S_obj 已会做事才能学到元认知，但 S_meta 的干扰又让 S_obj 学不会。因果图需要检测器先学会变量才能训转移。课程让每个模块在前置就绪后再启动。

### 损失函数

```
L = L_policy + 0.5×L_value + 0.03×H(π) + entropy_floor_penalty
  + 0.3 × (L_world + L_causal_detect
           [+P2: L_event_supervision×5 + L_var_causal×3 + L_causal_sparse×0.1])
  + 0.3 × L_interp
```

各损失：

| 损失 | 公式 | 适用阶段 | 频率 |
|------|------|---------|------|
| L_policy | REINFORCE: −Σ(log π(a) × (R−V).detach()) | P1+ | 每步 |
| L_value | MSE(V, normalized_returns) | P1+ | 每步 |
| H(π) | 策略熵奖励 | P1+ | 每步 |
| entropy_floor | max(0, 0.3−H) × 2.0 | P1+ | 每步 |
| L_world | MSE(world_model(h,a), h_next) | P1+ | 每步 |
| L_causal_detect | BCE(var_probs, ground_truth) | P1+ | 每步 |
| L_event_supervision | MSE(action_effects @ action_oh, gt_delta) × 5.0 | P2+ | 事件步 |
| L_var_causal | BCE(var_adj, co_occurrence_matrix) × 3.0 | P2+ | 事件步 |
| L_causal_sparse | L0.5(var_adj) × 0.1 | P2+ | 每步 |
| L_interp | MSE(interp, [loop,conf,expl,explt]) | P1+ | 每步 |

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
    {'params': causal_graph.action_effects,  'lr': 3e-4},
    {'params': causal_graph.var_causal_logits,'lr': 3e-4},
    {'params': meta_cog.parameters(),        'lr': 5e-5},
    {'params': meta_goal.parameters(),       'lr': 5e-5},
], eps=1e-5)

grad_clip = 0.5  # max_norm
```

### 训练配置

- 5 个不同种子的 CausalGridWorld 轮换，防过拟合
- GPU (CUDA) 和 CPU 均支持，自动检测
- 典型耗时：~300s(GPU) / ~960s(CPU)

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

## 完整迭代记录：8 轮实证

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

**加跳跃连接后第 4 轮结果**：CSM 69%, Baseline 43%。决定性突破。

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
| 结果 | interp 循环 0.37→0.19(递减!); action_effects 全≈0; 因果效应 0.429(假阳性) |
| 症状 | interp 前馈版用了 36 维输入，32 维 delta_S_obj 大部分为零淹没有效信号；action_effects 的 event_supervision 权重不够 |
| 病根 | delta_S_obj 占输入 32/36 维但 95% 时间全零 → 有效信号被淹没；sparsity 间接作用于 action_effects 通过梯度路径 |
| 修复 | interp 输入降到 3 个强信号(action_repeat, pred_err, entropy)；interp 网络加深(3→16→16→4)；action_effects sparsity 完全断开；action_effects 学习率提高到 3e-4 |

### 第 8 轮：因果图 action_effects 突破！var_adj 仍锁

| 项 | 值 |
|----|-----|
| 结果 | **pickup→has_key=0.249, use→door_open=0.278** (首次学到正确表征!); var_adj 仍全 0.451; CSM 74%; 因果效应 0.411 |
| 症状 | action_effects 正确但偏小；var_adj 完全未分化；因果效应 0.411 是假阳性(均匀传播) |
| 病根 | var_adj 的 sigmoid(logit) 仍卡在 0.5 均匀点——所有边同时接受相同方向梯度（预测损失的间接梯度），无法独立分化 |
| 修复 | 加 var_causal_supervision_loss（共现直接监督，BCE给每条边独立梯度）；先验初始化(has_key→door_open=2.0, saw_deco→others=-2.0)；var_causal_logits 学习率提高到 3e-4 |

**var_causal_supervision 和先验初始化的效果待验证。**

---

## 未解决的问题（优先级排序）

### P0: var_adj 仍全 ~0.45（因果图变量→变量边）

**现象**：`var_adj = sigmoid(var_causal_logits)` 全 ≈0.451，16 条边几乎相同

**已尝试但失败的方法**：
1. 间接梯度（predict_delta 的反传）→ 所有边接受相同方向梯度
2. sparsity (L0.5 范数) → 推向 0 但预测损失推向 0.5，平衡在 ~0.47
3. 事件步 10x 权重 → 仍不够

**v2.3 新增但待验证**：
4. `var_causal_supervision_loss`：BCE(var_adj, co_occurrence_matrix)——给每条边独立的方向
5. 先验初始化：logits[0,1]=2.0 (has_key→door_open 先验大)

**如果仍失败，建议尝试**：
- 完全断开 predict_delta 中的 mediated 路径，让 var_adj 只通过 var_causal_supervision 学习
- 将 var_adj 也改为直接参数化（不走 sigmoid），用 hard constraint 确保 [0,1]
- 用强化学习方式训练因果图：如果因果图正确预测了事件，给奖励
- 用环境提供干预数据：`env.intervene()` + `env.step()` 生成 do-calculus 数据

### P1: interp 对 action_repeat 响应不够

**现象**：重复 20 次同一动作时，循环警告从 0.45→0.43（应升到 >0.6）

**可能原因**：
1. interp 监督 target 的"循环"定义为 `prev_action == curr_action`，序列中大部分时候=1.0，网络学到输出恒定值
2. 更好的 target 应该是 `min(1.0, repeat_count / 5.0)`

**建议尝试**：
- 改变 interp 监督：`loop_target = min(1.0, action_repeat / 5.0)`
- 增大 interp_coef 从 0.3 到 1.0
- 确认 train.py 中 interp target 实际用的是 repeat count（当前代码可能仍用 `prev_a==curr_a`）

### P2: 元认知 delta_signal 对策略无可见影响

**现象**：P3 解锁后 delta_signal 开始调制 S_obj 重置门，但成功率无明显额外提升

**可能原因**：delta_signal 的幅度太小（×0.1 的偏置调制），对 SSM 行为几乎无影响

**建议尝试**：
- 将 delta_signal 的调制系数从 0.1 提到 0.5
- 给 delta_signal 加辅助监督：当 interp 检测到循环时，delta_signal 应该输出"增大重置"的信号
- 量化 delta_signal 的实际影响：对比 P3 开/关时 S_obj 的重置门激活值

### P3: 因果效应的假阳性

**现象**：do(has_key=1) → door_open=0.411，看似不错，但 var_adj 全 0.45 意味着**任何**变量的变化都会等量传播到所有变量

**这将在 var_adj 变得稀疏后自然解决。** 修复 P0 即修复 P3。

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
interp = FF_network(3_scalars_only)  # 不是 SSM_output, 不用 delta_S_obj

# 7. 事件步训练因果转移（不是每步训练！）
if is_event_step: train_causal_transfer()  # 5-10x 权重

# 8. 地面真值监督检测器
BCE(var_probs, ground_truth)  # 不是无监督

# 9. 共现监督 var_adj
BCE(var_adj, co_occurrence_matrix)  # 给每条边独立梯度，不是间接梯度

# 10. 学习进度型好奇心
curiosity = max(0, older_err - recent_err)  # 不是 raw prediction error

# 11. 熵地板
if entropy < 0.3: penalty += 2.0  # 防策略坍缩

# 12. 课程训练
P1: 只训核心; P2: 加因果; P3: 加元认知  # 不是一开始全开

# 13. GPU device 一致性
torch.tensor([...], device=tensor.device)  # 所有手动创建的 tensor

# 14. sparsity 只作用于 var_adj，不作用于 action_effects
sparsity_loss = L0.5(var_adj)  # 不是 L1(action_effects)
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

# ❌ interp 输入含大量零值维度
FF(delta_S_obj_32dim + 4_scalars)  # 32/36维噪声淹没有效信号

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

# ❌ var_adj 用间接梯度（predict_delta 反传）
# 所有边接受相同方向梯度→无法分化→均匀0.47

# ❌ 因果效应假阳性判断
# var_adj 全0.45时任何干预都会等量传播到所有变量，不是真因果
```

---

## 文件结构

```
csm_v2/
├── device_utils.py      # GPU/CPU 自动检测
├── env.py               # 因果网格世界（含地面真值、干预接口）
├── modules.py           # 所有神经网络模块定义
├── csm_agent.py         # CSM v2.2 完整智能体
├── train.py             # 课程训练循环
└── evaluate.py          # 因果推断 + 元认知可视化
```

---

## 运行

```bash
pip install torch numpy matplotlib

# 训练 (~300s GPU / ~960s CPU)
python train.py

# 因果推断测试（检查 action_effects 和 var_adj）
python evaluate.py cf

# 元认知可解释性（检查 interp 随 repeat 的响应）
python evaluate.py interp
```

---

## 模块当前状态

| 模块 | 功能 | 验证数据 | 状态 |
|------|------|---------|------|
| Encoder + SSM + ActionHead | 核心路径 | 74% 成功率 | ✅ 可靠 |
| 跳跃连接 | 信息生命线 | 移除后暴跌 | ✅ 可靠 |
| 因果图检测器 | 检测4变量 | v_key≈0.83, v_door≈0.62 | ✅ 地面真值监督有效 |
| 因果图动作效果 | 动作→变量 | pickup→key=0.25, use→door=0.28 | ✅ **首次突破!** |
| 因果图变量因果 | 变量→变量 | var_adj 全≈0.45 | ❌ 未分化 |
| 场景路由 | 加法增强 | 初始化为恒等映射 | ✅ 可靠 |
| 元认知 interp | 可解释信号 | 循环 0.45→0.43 (响应弱) | ⚠ 前馈解决不动点，响应不够 |
| 元认知 delta_signal | SSM调制 | 调制 S_obj 重置门 ×0.1 | ⚠ 待验证实际影响 |
| 槽位记忆 | 工作记忆 | 短 episode 难评估 | ⚠ 存在 |
| 内部奖励 | 驱动探索 | 学习进度型，封顶 0.5 | ✅ 可靠 |

---

## 下一步（按优先级）

1. **验证 var_causal_supervision + 先验初始化**：当前代码已加入但未跑完，这是 var_adj 分化的最可能路径
2. **修复 interp 监督 target**：从 `prev_a==curr_a` 改为 `min(1.0, repeat/5.0)`，与 action_repeat 强度正相关
3. **量化 delta_signal**：测量 P3 开/关时 S_obj 重置门激活的差异，确认调制是否真的有效
4. **多房间环境**：CSM 的架构优势在长序列复杂任务中才会真正显现（80步→500步）
5. **睡眠/巩固机制**：推理时记录高奖励轨迹，周期性回放更新参数
6. **时间对齐层**：多模态输入时间尺度差异需要对齐

---

## 诚实的结论

74% vs 43% 证明了**架构方向正确**。第 8 轮因果图 action_effects 的突破证明了：**即使是最顽固的模块，只要找到正确的监督信号和参数化方式，也能学到有意义的表征**。

关键教训：
- **间接梯度是因果图的死敌**——所有边接受相同方向的梯度，无法分化。必须用直接监督给每条边独立的方向。
- **sigmoid 是稀疏性的敌人**——sigmoid(0)=0.5 是梯度甜蜜点，模型会卡在那里。直接参数化或硬先验是出路。
- **信息流不能有串联瓶颈**——跳跃连接是最重要的单项改进。
- **课程训练是解决鸡生蛋问题的唯一方式**——不能一开始就全部解锁。

但还没有证明每个模块都在独立贡献。跳跃连接做了大部分重活，高级模块正在从"不拖后腿"向"独立贡献"过渡。

真正的分水岭在多房间环境：当 episode 从 80 步变成 500 步，当需要跨房间记住钥匙位置，当需要在多个子目标间切换——那时工作记忆、因果推理、元认知才会从"没拖后腿"变成"不可或缺"。

---

## 附录：可验证的数字

| 指标 | CSM v2.2 | PPO+LSTM Baseline |
|------|---------|------------------|
| 参数量 | 46,524 | 89,959 |
| 最终成功率 | 74% | 43% |
| 最终平均奖励 | 1.056 | 0.362 |
| P1 结束时成功率 | 31% | ~25% |
| P2 结束时成功率 | 38% | ~35% |
| P3 峰值成功率 | 74% | ~50% |
| 训练耗时(CPU) | ~960s | ~300s |
| action_effects[pickup,has_key] | 0.249 | N/A |
| action_effects[use,door_open] | 0.278 | N/A |
| var_adj (所有边) | ≈0.451 | N/A |
| 因果效应 do(key→door) | 0.411(假阳性) | N/A |
| interp 循环响应 | 0.45→0.43(弱) | N/A |
| var_key 检测精度 | 0.83 | N/A |
| var_door 检测精度 | 0.62 | N/A |

# CSM v3.1 — 概念状态机实证优化架构

> 自包含文档。发给任何 LLM，它应能理解完整架构、复现代码、继续调试未解决模块、或扩展到新环境。

---

## 一句话总结

CSM 是层次化认知架构，用显式状态机替代 Transformer 隐式注意力。v3.1 引入 GrowableStateSpace（可扩充概念池）和 LowRankDeltaA（低秩元认知调制），在 7×7 因果网格世界上达到 **74% 成功率**。循环信号首次正确递增（0.074→0.896），因果效应稳定（0.705）。当前瓶颈：概念池增长过快，ΔA影响微弱，成功率较v2.2基线（79%）回退5%。

---

## 架构信息流

```
obs (56维)
    │
    ▼
┌──────────┐
│ Encoder  │ 56→64, 2层MLP+GELU
└────┬─────┘
     │ h (64维)──────────────────────┐ (跳跃连接)
     │                               │
     ├──────┬────────────┐           │
     ▼      ▼            ▼           │
┌───────┐ ┌──────┐ ┌──────────┐     │
│ C₋₁   │ │因果图│ │场景路由   │     │
│ 元目标 │ │  G   │ │加法增强   │     │
│ 8维goal│ │4变量  │ │h+α·enh  │     │
└───┬───┘ └──┬───┘ └────┬─────┘     │
    │        │          │            │
    └────────┤   ┌──────────────┐    │
             │   │  S_obj       │    │
             │   │  GatedSSM    │◀───│── LowRankΔA 调制(r×0.5 + S×0.2)
             │   │  32维状态     │    │
             │   └──────┬───────┘    │
             │          │            │
             │   ┌──────────────┐    │
             │   │ 概念池(可扩充) │    │
             │   │ L2距离写入    │    │
             │   │ 注意力读取32维│    │
             │   └──────┬───────┘    │
             │          │            │
             │   ┌──────────────┐    │
             │   │  槽位记忆     │    │
             │   │  4×64维      │    │
             │   └──────┬───────┘    │
             │          │            │
             ▼          ▼            ▼
        ┌────────────────────────────┐
        │  c2 = Linear(S_obj,        │
        │   concept_read, slot, h)   │
        │  32+32+64+64 = 192 → 64   │
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
     LowRankΔA     interp(4维)   目标更新
     (秩4,          循环:独立头    信号
      16→4→32)      abs(s)*r+b
      sigmoid(gate)  共享:3标量
                    →16→3
```

---

## 各模块完整规格

### 1. Encoder (56→64)
```python
nn.Sequential(nn.Linear(56, 64), nn.GELU(), nn.Linear(64, 64))
```

### 2. GatedSSM (实证修正: 替代裸递归)
```python
z = sigmoid(W_z(concat(x, S)))         # 遗忘门
r = sigmoid(W_r(concat(x, S)))         # 重置门
if delta_A: r = clamp(r + delta_A * 0.5, 0, 1)  # ΔA调制(0.3→0.5增强)
h_hat = tanh(W_h(concat(x, r * S)))
S_new = (1-z)*S + z*h_hat
if delta_A: S_new = S_new + delta_A * 0.2          # 状态扰动(0.1→0.2增强)
```
**铁律**: 裸递归 `S'=A·S+B·x` 必崩溃(梯度爆炸/消失), (1-z)∈[0,1]保证稳定。

### 3. LowRankDeltaA (新增: 替代全秩ΔA和偏置门控)
```python
class LowRankDeltaA:
    down: Linear(16, 4)    # 压缩
    up: Linear(4, 32)      # 展开, init normal(0, 0.01)
    norm_gate: Linear(16, 1) # 幅度控制
    forward: delta = up(relu(down(S_meta))) * sigmoid(norm_gate(S_meta))
```
**当前问题**: 幅度约0.003, Δr约0.001, 影响微弱。学习率3e-4, 但P3仅1400ep可能不够。

### 4. 场景路由加法增强 (实证修正: 替代乘法掩码)
```python
enhancement = Σ(weights_i × enh_vector_i)  # 16个场景原型
alpha = sigmoid(gate(h, goal))             # 门控强度
h_out = h + alpha * enhancement            # 加法!初始化退化为恒等映射
```
**铁律**: 乘法掩码 `h *= mask` 丢弃85%信息不可恢复。

### 5. 跳跃连接 (最大单项改进)
```python
c2 = Linear(S_obj(32), concept_read(32), slot_read(64), h(64))  # 192→64
```
**铁律**: h必须在c2输入中! 移除后成功率暴跌。

### 6. 因果图 G (稳定, 未改动)
- **检测器**: BCE(var_probs, ground_truth), 地面真值监督
- **action_effects**: 直接参数化(不走sigmoid!), 事件步5x监督
- **var_adj**: 先验初始化(logit[0,1]=2.0) + 共现监督BCE, sparsity仅var_adj
- **反事实**: do(has_key=1)→door_open=0.705 (真实因果)

**实证结果**:
```
action_effects: pickup→has_key=0.202, use→door_open=0.268
var_adj: has_key→door_open=0.806, saw_deco→*=0.072
因果效应: 0.705
```

### 7. 可解释信号 interp (v3.1突破: 循环信号修复)
```python
# 独立循环头: 数学保证单调递增
loop_logit = abs(scale) * action_repeat + bias  # abs()强制scale为正!
loop = sigmoid(loop_logit).unsqueeze(0)

# 共享头: 置信/探索/利用
shared = interp_shared(pred_err, confidence, entropy)  # 3→16→3→sigmoid
```
**训练**: 循环头用BCE with logits + 加权(高repeat样本3x), 共享头用MSE。
**结果**: 循环信号 0.074→0.896 (v2.2是0.086→0.039方向反转!)

### 8. GrowableStateSpace (新增: 可扩充概念池)
```python
class GrowableStateSpace:
    query_proj: Linear(64, 32)
    new_entry_proj: Linear(64, 32)
    update_gate: Linear(32, 1)
    seed: Parameter(4, 32)
    dist_threshold: 1.5  # L2距离阈值

    read(query, pool):  # 注意力→固定32维
        attn = softmax(pool @ q / sqrt(32))
        return (attn * pool).sum(0)

    write(query, pool, threshold):  # L2距离判断
        dist = (pool - q).norm(dim=-1)
        if min_dist < threshold: 软更新最匹配条目
        else: 追加新条目到池尾

    concentration_loss(query, pool):  # 防止投影太散
        min_dist = (pool - q).norm(dim=-1).min()
        return min_dist * 0.1
```
**当前问题**: P2+阈值1.0太低, 池增长过快(eval中40步从5长到24)。已修: 阈值改2.0/3.0 + 加concentration_loss。

### 9. S_meta 元认知层
```python
# 输入: ΔS_obj(32) + repeat(1) + pred_err(1) + confidence(1) + entropy(1) = 36维
trace_encoder: Linear(36, 16)
meta_ssm: GatedSSM(16, 16)
# 输出1: LowRankDeltaA → 调制S_obj
# 输出2: interp(独立循环头 + 共享3信号)
```
**铁律**: 输入必须ΔS_obj(非绝对值!), interp必须前馈(不走SSM), 循环信号必须独立头。

### 10. 内部奖励
```python
R_internal = 0.5 × learning_progress + 0.3 × info_gain
R_internal = min(R_internal, 0.5)  # 封顶
# 学习进度 = max(0, 旧10步平均误差 - 新10步平均误差) (非原始误差!)
```

---

## 训练范式

### 三阶段课程
| 阶段 | Episodes | 活跃 | 冻结 |
|------|----------|------|------|
| P1 核心 | 0–799 | Encoder, SSM, ActionHead, WorldModel, Slot, 检测器, 概念池(阈值3.0) | S_meta(ΔA=0), goal(×0), 因果转移 |
| P2 因果 | 800–1599 | +因果转移(事件步5x), +共现监督, +概念池(阈值2.0) | S_meta(ΔA=0), goal(×0) |
| P3 元认知 | 1600+ | 全部: ΔA解锁, goal解锁 | 无 |

### 优化器
```python
optimizer = Adam([
    {'params': encoder, 'lr': 3e-4},
    {'params': obj_ssm, 'lr': 3e-4},
    {'params': c2_proj, 'lr': 3e-4},
    {'params': action_head, 'lr': 3e-4},
    {'params': slot_mem, 'lr': 3e-4},
    {'params': world_model, 'lr': 3e-4},
    {'params': scene_router, 'lr': 1e-4},
    {'params': concept_pool, 'lr': 1e-4},
    {'params': causal_graph.var_detector, 'lr': 1e-4},
    {'params': causal_graph.action_effects, 'lr': 3e-4},
    {'params': causal_graph.var_causal_logits, 'lr': 3e-4},
    {'params': meta_cog.low_rank_delta_a, 'lr': 3e-4},  # 独立高学习率!
    {'params': meta_cog.trace_encoder, 'lr': 5e-5},
    {'params': meta_cog.meta_ssm, 'lr': 5e-5},
    {'params': meta_cog.loop_scale, 'lr': 3e-4},   # 单调循环头
    {'params': meta_cog.loop_bias, 'lr': 3e-4},
    {'params': meta_cog.interp_shared, 'lr': 1e-4},
    {'params': meta_goal, 'lr': 5e-5},
], eps=1e-5)
grad_clip = 0.5
```

### 损失函数
```
L = L_policy + 0.5×L_value + 0.03×H(π) + max(0, 0.3-H)×2.0
  + 0.3 × (L_world + L_causal_detect + L_pool_concentrate)
  + 0.3 × (P2: L_event×5 + L_var_causal×3 + L_sparse×0.1)
  + 0.3 × L_interp_shared(MSE)
  + 1.0 × L_loop_bce(BCE_with_logits, 高repeat加权3x)
```

---

## 环境：7×7 因果网格世界

```
┌───────────────┐
│ . . . │ . . T │
│ . K . │ . . . │  K=钥匙 D=门 T=宝藏 5=装饰(干扰)
│ . . . D . . . │
│ . . . │ . . . │
│ . A . │ . . 5 │
└───────────────┘
```
- obs: 3×3视野(6类×9格=54) + has_key + door_open = 56维
- 6动作: up/down/left/right/pickup/use
- 塑形奖励: 靠近子目标+0.1, 捡钥匙+0.5, 开门+0.5, 宝藏+1.0
- 地面真值: [has_key, door_open, near_treasure, saw_deco]

---

## 设计铁律 (9轮+实证换来的禁忌)

### 架构层
- ❌ 裸递归 S'=A·S+B·x → 必须门控 (1-z)S+z·tanh(...)
- ❌ 乘法掩码 h*=mask → 必须加法增强 h+α·enhancement
- ❌ 无跳跃连接 → h必须直通c2
- ❌ S_meta输入S_obj绝对值 → 必须ΔS_obj
- ❌ interp走SSM → 必须前馈
- ❌ interp多输出共享网络 → 循环必须独立头
- ❌ loop_head用MLP → 必须abs(scale)*x+bias 保证单调

### 训练层
- ❌ 每步训因果转移 → 必须事件步5-10x
- ❌ sparsity作用于action_effects → 仅对var_adj
- ❌ 检测器无地面真值 → 必须BCE监督
- ❌ var_adj用间接梯度 → 必须共现+先验
- ❌ action_effects走sigmoid → 必须直接参数化从0开始
- ❌ 全模块同时训练 → 必须课程
- ❌ 好奇心用原始误差 → 必须学习进度
- ❌ 奖励只有稀疏终点 → 必须塑形

### 代码层
- ❌ CSMv3Agent()不to(DEVICE) → 必须CSMv3Agent().to(DEVICE)
- ❌ 概念池用余弦相似度 → 必须L2距离
- ❌ 概念池无集中约束 → 必须concentration_loss
- ❌ BCE无样本加权 → 高repeat必须3x加权

---

## 当前未解决问题 (优先级排序)

### P0: 概念池增长过快
- **现象**: eval中40步从5长到24条, 训练中P2长到84条
- **已修(待验证)**: L2阈值从1.0→2.0/3.0 + concentration_loss
- **根因**: query_proj把不同观测投影得太散, L2距离总是>阈值

### P1: ΔA影响微弱
- **现象**: ΔA_mag≈0.003, Δr≈0.001, 对S_obj几乎无影响
- **已修(待验证)**: 调制系数0.3→0.5, 扰动0.1→0.2
- **可能下一步**: P3阶段延长到3000ep, 或ΔA直接乘S(替代加偏置)

### P2: 成功率回退 79%→74%
- **现象**: v3.1比v2.2低5%
- **可能原因**: 概念池注入噪声(P0的后果), 或c2_proj从160→192维增加了拟合难度
- **消融实验**: 跑无概念池的v3.1, 对比确认池是否拖后腿

### P3: var_adj方向性仍依赖先验
- door_open→has_key=0.357仍偏高, 共现无法区分方向
- 需要: 时序偏置("i先变j后变→i→j更强")

---

## 完整迭代记录摘要

| 轮次 | 关键事件 | 成功率 |
|------|---------|--------|
| 1 | 裸递归崩溃→门控SSM | 12% |
| 2 | PPO ratio=1假更新→REINFORCE | 6% |
| 3 | 奖励太稀疏→塑形奖励 | 12%→29% |
| 4 | 乘法掩码瓶颈→加法增强+跳跃连接 | **69%** |
| 5-6 | 因果图均匀0.47→事件步+直接参数化 | 69% |
| 7 | interp梯度绑架→独立头思路 | 74% |
| 8 | action_effects突破, var_adj仍锁 | 74% |
| 9 | var_adj先验+共现突破, 因果效应0.699 | **79%** |
| v3 | LowRankDeltaA+GrowableStateSpace | 66%→74% |
| v3.1 | 单调循环头修复, 概念池L2+集中 | 74%(待验证修复) |

---

## 文件清单

```
csm_v3.1/
├── device_utils.py      # GPU/CPU自动检测, FORCE_DEVICE="cpu"/"cuda"
├── env.py               # 因果网格世界(含地面真值)
├── modules.py           # GatedSSM, LowRankDeltaA, MetaCognition, CausalGraph等
├── growable_state.py    # GrowableStateSpace(L2距离+concentration_loss)
├── csm_agent.py         # CSMv3Agent(含跳跃连接, 概念池阈值2.0/3.0)
├── train.py             # 三阶段课程训练(REINFORCE+BCE加权)
├── evaluate.py          # cf/interp/pool/da 四种评估
└── README.md            # 本文档
```

---

## 运行

```bash
python train.py                    # 训练 ~1100s CPU
python evaluate.py cf              # 因果推断测试
python evaluate.py interp          # 可解释信号(循环应0.07→0.89递增)
python evaluate.py pool            # 概念池增长轨迹
python evaluate.py da              # ΔA幅度与影响量化
```

---

## 下一步方向

1. **验证P0/P1修复**: 概念池concentration_loss+阈值提高是否减缓增长, ΔA系数加倍是否增强影响
2. **消融实验**: 无概念池的v3.1 vs 有概念池, 确认池是否拖后腿
3. **ΔA直接乘S**: 替代加偏置, `S_new = (1-z)*S + z*tanh(W_h·[x, r*(S + β·ΔA)])`, 更接近原始构想
4. **多房间环境**: MiniGrid MultiRoom, 500步, CSM架构优势应放大
5. **时序共现**: var_adj方向性脱离先验依赖
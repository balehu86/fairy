# CSM v2.1 概念状态机架构

## 概述

CSM (Conceptual State Machine) 是一种层次化认知架构。核心思想：用显式状态机结构替代 Transformer 的隐式注意力，通过场景感知路由实现稀疏激活，通过双层 SSM 实现元认知，通过因果图实现反事实推理。

v2.1 经历 5 轮实证迭代。v2.0 在 7x7 因果网格世界上仅 12% 成功率（PPO+LSTM baseline 为 43%），修复后达 69%。

---

## 架构总览

```
输入 obs (56维)
    |
    v
[Encoder] ---------> h (64维) ----------------------------+
    |                                                      |
    +----------+-----------+                               |
    v          v           |                               |
[C-1元目标] [因果图G]       |                               |
  8维goal    4离散变量      |                               |
    |          |           |                               |
    +----+-----+           |                               |
         |                 |                               |
         v                 |                               |
    [C0 抽象概念层]         |                               |
         |                 |                               |
         v                 |                               |
    [场景路由]              |                               |
    h_out=h+a*enhance      |                               |
      |       |            |                               |
      v       v            |                               |
    [S_meta] [S_obj]       |                               |
    元认知层  GatedSSM      |                               |
    输入:dS_obj            |                               |
    输出:Dr偏置             |                               |
      |       |            |                               |
      +---+---+            |                               |
          |                |                               |
          v                |                               |
    [概念状态+槽位记忆]      |                               |
    c2=proj(S_obj,slot,h) <+-------------------------------+
          |
          v
    [ActionHead] -> action_probs, value
```

---

## 各模块详述

### 1. Encoder (56 -> 64)

两层 MLP + GELU。将部分可观测 3x3 视野 one-hot (54维) + 内部状态 (2维: has_key, door_open) 编码为 64 维 h。

### 2. C-1 元目标层 (8维)

输入: h(64维) + 3个标量信号

输出: 8维目标向量 goal = tanh(Linear(concat(h, signals)))

三个信号:

- 好奇心: 基于学习进度(预测误差下降率)，不是原始预测误差。原始误差会被噪声黑入，学习进度只奖励"正在变得可预测的新颖性"
- 能力缺口: 近期负奖励的统计
- 外部奖励: 当步环境奖励

课程控制: 阶段1-2时 goal 乘以0(冻结)，不干扰核心学习。

### 3. 因果图 G

4个离散变量: has_key, door_open, near_treasure, saw_decoration

saw_decoration 是干扰变量(与任务无关)，测试系统能否区分真因果和虚假相关。

**检测器**: sigmoid(Linear(h)) -> 4个变量概率

- 用环境地面真值做 BCE 监督(关键，无监督则锁死0.5)
- 已验证: v_key=0.833, v_door=0.620

**转移预测**: predict_delta(var_probs, action_one_hot)

- 预测变量的变化量(tanh输出)，不是绝对值
- 结构方程: struct_eq_i(parents * A[:,i] concat action_oh) -> delta_v_i
- 为什么预测增量: 预测绝对值时偏置约0.5即可最小化损失，A矩阵完全没有梯度。预测增量时偏置无法解释变化，A必须学

**稀疏约束**: L0.5范数，比L1更强地把弱连接推向精确的0

**反事实**: counterfactual(var_probs, intervene_idx, intervene_val)

- do(X=x)操作: 固定干预变量值，切断其入边，用修改后的值预测下游

### 4. 场景感知稀疏路由

这是 v2.0 到 v2.1 最关键的修复之一。

v2.0 方案:

```
h_out = h * sparse_mask    # 乘法掩码
# 保留top 15%维度，85%信息被永久删除
# 路由未训好 -> 信息丢失 -> 下游全崩 -> 12%成功率
```

v2.1 方案:

```
enhancement = sum(weights_i * enhancement_vector_i)    # 加权增强
alpha = sigmoid(gate(concat(h, goal)))                  # 门控强度
h_out = h + alpha * enhancement                         # 加法增强
# 未训练时 alpha约0, enhancement约0, h完整保留(退化为直通)
```

16个场景原型，每个有可学习的增强向量。原型与h计算余弦相似度，加上目标偏置，softmax得权重。

### 5. S_obj 对象层 GatedSSM (state_dim=32)

v2.0 用裸A矩阵递归，特征值稍大于1就梯度爆炸，小于1就消失。

v2.1 用GRU门控结构(SSM的稳定离散化):

```
z = sigmoid(W_z * concat(x, S))         # 遗忘门
r = sigmoid(W_r * concat(x, S) + Dr)    # 重置门(受S_meta调制)
S_new = (1-z)*S + z*tanh(W_h * concat(x, r*S))
y = output_proj(S_new)
```

Dr由S_meta提供，调制重置门偏置(不是直接改A矩阵)。

### 6. S_meta 元认知层 (state_dim=16)

核心原则: S_meta 不处理外部输入，只处理 S_obj 的推理过程本身。

v2.0 问题: 输入S_obj绝对值 -> S_obj步间几乎不变 -> 输入恒定 -> 不动点 -> interp锁在0.5

v2.1 修复: 输入增量 dS_obj = S_obj(t) - S_obj(t-1)

- 绝对值不变不代表增量也为零
- 增量捕捉"推理过程正在发生什么变化"

内部也是GatedSSM，输出:

1. Dr调制信号: 传给S_obj的重置门偏置，修改S_obj的"思维方式"
2. interp可解释信号(4维, sigmoid):
   - interp[0] 循环警告: target = (a_t == a_{t-1})
   - interp[1] 置信度: target = 1 - pred_err
   - interp[2] 需要探索: target = min(pred_err * 3, 1)
   - interp[3] 需要利用: target = (r > 0)
   - 用 MSE(interp, target) 做辅助监督 -- "自指"系统仍需外部锚点

课程控制: 阶段1-2时Dr乘以0，不干扰S_obj学习。

### 7. 槽位工作记忆 (4槽 x 64维)

解决SSM类模型长程精确召回弱的问题:

- 读: 注意力 attn = softmax(slots @ query / sqrt(d))，加权求和
- 写: 软门控 slots += write_weights * (query - slots) * 0.3
- 初始化为可学习的空token

### 8. 层次概念状态与跳跃连接

```
c2 = Linear(concat(S_obj(32), slot_read(64), h(64)))  -> 64维
                                              ^
                                         跳跃连接!
```

这是12%到69%的最大贡献者。ResNet洞见: 深层模块应该是残差/增强，不是替代。h通过跳跃连接直通到动作头，即使中间模块全部失败，信息仍有通路。

### 9. 内部奖励

```
R_intrinsic = 学习进度 * 0.5 + 信息增益 * 0.3
```

- 学习进度: max(0, 旧预测误差 - 新预测误差)，封顶0.5防止失控
- 信息增益: 因果变量置信度的平均绝对变化
- 不使用原始预测误差(会被不可预测噪声黑入)

### 10. 控制符系统

时间节律 / 导航 / 结构 / 生成。当前简化版本未显式实现，预留接口。

---

## 训练范式

### 课程三阶段

| 阶段 | Episode | 活跃模块 | 冻结模块 |
|------|---------|---------|---------|
| P1 核心 | 0-799 | Encoder, SSM, ActionHead, WorldModel, Slot, 因果检测器 | S_meta干预, 目标调制, 因果转移 |
| P2 +因果 | 800-1599 | +因果转移预测, 场景路由目标调制 | S_meta干预 |
| P3 +元认知 | 1600+ | 全部解锁 | 无 |

### 损失函数

P1:

```
L = L_policy(REINFORCE) + 0.5*L_value + L_world + L_causal_detect(BCE) + 熵地板
```

P2 增加:

```
+ L_causal_predict(MSE, predict_delta) + 0.5*L_causal_sparse(L0.5)
```

P3 增加:

```
+ 0.2*L_interp(MSE, 地面真值target) + 内部奖励
```

### 训练细节

- 优化器: Adam，不同模块不同学习率
  - 核心(Encoder/SSM/ActionHead): lr=3e-4
  - 路由/因果: lr=1e-4
  - 元认知/目标: lr=5e-5
- 梯度裁剪: max_norm=0.5
- 熵地板: 0.3 (低于此值加惩罚，防止策略坍缩到单动作)
- 5个环境种子轮换防过拟合
- 单episode REINFORCE (非PPO批量，因CPU速度限制)

---

## 环境设计

7x7因果网格世界:

- 左侧有钥匙(K)，中间有门(D)，右侧有宝藏(T)
- 因果链: has_key=True -> can_open_door=True -> can_reach_treasure=True
- 干扰: 随机装饰符号(5)与任务无关，测试因果vs相关
- 部分可观测: 3x3视野 + 内部状态
- 6个动作: up/down/left/right/pickup/use
- 塑形奖励: 靠近当前子目标+0.1/步，拿钥匙+0.5，开门+0.5，宝藏+1.0
- 地面真值接口: get_ground_truth() 返回 [has_key, door_open, near_treasure, saw_deco]

---

## 迭代教训 (最重要)

**1. 裸递归SSM不可训练**

S'=A*S+B*x 在长序列上梯度爆炸/消失。必须用门控结构(GRU/LSTM/Mamba选择性机制)。

**2. 乘法掩码=信息瓶颈**

h*mask 在mask未训好时丢弃大部分信息。必须用加法增强 h+Dh，使模块退化为恒等映射时信息无损。

**3. 跳跃连接是深层架构的生命线**

每个中间模块都可能失败，跳跃连接保证信息有旁路。没有跳跃连接，架构复杂度是纯负债。

**4. 预测绝对值无法学到结构**

当target约等于input时，偏置即可解，结构参数(如A矩阵)完全没有梯度。必须预测增量/变化量。

**5. 不变输入导致不动点**

S_obj步间变化微小，S_meta若输入绝对值则锁死。输入增量dS_obj才能捕捉动态。

**6. 自指系统仍需外部锚定**

interp无监督则锁在0.5。必须用可观测的外部信号做辅助target。自指不等于无锚定。

**7. 课程训练解除鸡生蛋**

高级模块(S_meta)需要低级模块(S_obj)先学会，但高级模块的干扰阻止低级模块学习。分阶段解锁是唯一解法。

**8. 单episode REINFORCE高方差但可用**

PPO批量需要更多计算。REINFORCE + 标准化returns + 熵地板在小规模上工作。

**9. 地面真值监督是桥梁**

纯自监督在复杂架构中几乎不可能收敛。用环境地面真值监督检测器，等检测器稳定后再训转移预测。

**10. 好奇心必须用学习进度**

原始预测误差会被不可预测噪声黑入。学习进度 max(0, old_err-new_err) 只奖励"正在变得可预测"的新颖性。

---

## 当前状态与待验证

| 模块 | 状态 | 验证情况 |
|------|------|---------|
| 核心路径 (Encoder+SSM+Action) | 正常工作 | 69%成功率 |
| 跳跃连接 | 正常工作 | 12%->69% |
| 因果检测器 | 正常工作 | v_key=0.833 |
| 因果转移预测 | 修复中 | predict_delta+动作条件化，待跑结果 |
| 元认知interp | 修复中 | dS_obj+辅助监督，待跑结果 |
| 场景路由 | 正常工作 | 加法增强，gate已训出有效alpha |
| 槽位工作记忆 | 存在 | 在简化任务上难以单独评估 |
| 内部奖励 | 正常工作 | 学习进度型，未出现噪声黑入 |

下一步: 在多房间/更复杂环境上验证因果转移和元认知是否真正带来样本效率优势。

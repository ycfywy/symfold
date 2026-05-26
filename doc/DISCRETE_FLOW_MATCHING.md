# Discrete Flow Matching 详解

> 本文通过一个**完整的数值例子**，从训练到推理，逐步拆解 SymFold 中 Discrete Flow Matching 的全部计算过程。

---

## 一、核心直觉

RNA 二级结构预测的输出是一个 **L×L 的对称二值矩阵**（Contact Map）：

```
RNA 序列:  A U G C G C A U  (L=8)
配对:      (1,8) (2,7) (3,6)

Ground Truth Contact Map x₁:
    1 2 3 4 5 6 7 8
1 [ 0 0 0 0 0 0 0 1 ]
2 [ 0 0 0 0 0 0 1 0 ]
3 [ 0 0 0 0 0 1 0 0 ]
4 [ 0 0 0 0 0 0 0 0 ]
5 [ 0 0 0 0 0 0 0 0 ]
6 [ 0 0 1 0 0 0 0 0 ]
7 [ 0 1 0 0 0 0 0 0 ]
8 [ 1 0 0 0 0 0 0 0 ]
```

**问题**：如何训练一个模型来预测这个矩阵？

**SymFold 的答案**：不直接回归，而是**学习一个"从噪声逐步生成"的过程**。

---

## 二、什么是 Discrete Flow Matching？

### 2.1 对比连续扩散模型

| | 连续扩散 (DDPM/Score Matching) | **离散流匹配 (SymFold)** |
|---|---|---|
| 数据类型 | 连续值 (图像像素 0~255) | **二值 {0, 1}** |
| 噪声过程 | 加高斯噪声 | **Bernoulli 插值** |
| 前向公式 | x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε | **x_t ~ Bernoulli(t·x₁ + (1-t)·ρ₀)** |
| 模型预测 | 预测噪声 ε 或 score ∇logp | **预测"干净"目标 P(x₁=1)** |
| 采样方式 | DDIM/DDPM 去噪 | **τ-leap CTMC 翻转 bits** |
| 适用场景 | 图像、音频 | **二值矩阵、离散图** |

### 2.2 为什么对 RNA 结构用离散？

RNA Contact Map 的每个位置只有两种状态：**配对 (1)** 或 **不配对 (0)**。用高斯扩散意味着：
- 需要在 [0,1] 连续空间建模，然后四舍五入
- 阈值选择敏感，且"0.49 vs 0.51"不应该有显著意义差异

Discrete Flow Matching 直接在 {0, 1} 空间工作：
- **物理上自然**：配对就是配对，没有"半配对"
- **先验好定义**：背景噪声 = 极稀疏的随机 1（ρ₀=0.005 ≈ 真实数据的配对率）
- **采样是 bit-flip**：每步只决定"翻还是不翻"

---

## 三、训练过程：从头到尾的数值例子

### 前提设定

假设我们有一条 L=8 的 RNA，ground truth contact map 如上。

**模型配置**：
- `ρ₀ = 0.005`（先验噪声率，≈ 真实数据中"1"的比例）
- `pos_weight_base = 199`
- `pos_weight_min = 20`
- `focal_gamma = 1.5`

---

### Step 1: 采样时间 t

```python
t = torch.rand(1)  # 假设采样到 t = 0.6
```

时间 t 代表"信号占比"：
- t = 0 → 纯噪声（几乎全 0）
- t = 1 → 完整 ground truth
- **t = 0.6 → 60% 信号 + 40% 噪声**

---

### Step 2: 构造噪声样本 x_t

**公式**：对每个位置 (i,j)，独立采样：

```
P(x_t[i,j] = 1) = t × x₁[i,j] + (1-t) × ρ₀
```

对于 t=0.6：

**情况 A：GT 为 1 的位置（如 (1,8)）**
```
P(x_t[1,8] = 1) = 0.6 × 1 + 0.4 × 0.005 = 0.602
```
→ 有 60.2% 的概率保留为 1

**情况 B：GT 为 0 的位置（如 (1,2)）**
```
P(x_t[1,2] = 1) = 0.6 × 0 + 0.4 × 0.005 = 0.002
```
→ 只有 0.2% 的概率变成 1（噪声）

**实际操作**：对每个位置，生成 rand < p_one 即为 1：

```python
p_one = t * x_1 + (1-t) * rho_0
x_t = (torch.rand_like(x_1) < p_one).float()
```

**假设这次采样得到的 x_t**（t=0.6）：
```
x_t:
    1 2 3 4 5 6 7 8
1 [ 0 0 0 0 0 0 0 1 ]  ← (1,8) 保留了
2 [ 0 0 0 0 0 0 1 0 ]  ← (2,7) 保留了
3 [ 0 0 0 0 0 0 0 0 ]  ← (3,6) 这次没保留！（40%概率丢失）
4 [ 0 0 0 0 0 0 0 0 ]
5 [ 0 0 0 0 0 1 0 0 ]  ← 噪声！(5,6)=1 是假的
6 [ 0 0 0 0 1 0 0 0 ]  ← 对称噪声
7 [ 0 1 0 0 0 0 0 0 ]
8 [ 1 0 0 0 0 0 0 0 ]
```

**对称化**：取 `max(x_t, x_t^T)` 确保 (i,j) 和 (j,i) 一致。

---

### Step 3: 模型前向（预测 logit）

模型输入：`(x_t, t=0.6, RNA序列条件)`

模型输出：`logit[i,j]` — 对每个位置预测"这里应该是 1 的对数几率"。

**假设模型输出的 logit**：
```
logit (简化只看关键位置):
  位置 (1,8): logit = +4.2  → P(x₁=1) = sigmoid(4.2) = 0.985   ✓ 高确信配对
  位置 (2,7): logit = +3.8  → P(x₁=1) = sigmoid(3.8) = 0.978   ✓ 高确信配对
  位置 (3,6): logit = +2.1  → P(x₁=1) = sigmoid(2.1) = 0.891   ✓ 有点不确定但还是对的
  位置 (5,6): logit = -1.5  → P(x₁=1) = sigmoid(-1.5) = 0.182  ✓ 识破了噪声
  位置 (4,5): logit = -6.0  → P(x₁=1) = sigmoid(-6.0) = 0.002  ✓ 确定不配对
```

---

### Step 4: 计算 Loss

#### 4.1 基础 BCE

```python
# 对每个位置计算 binary cross-entropy
# GT x_1[i,j], 模型输出 logit[i,j]
bce[i,j] = -(pos_weight × x₁ × log(σ(logit)) + (1-x₁) × log(1-σ(logit)))
```

**核心难题**：数据极度不平衡！

- L=8 时，矩阵有 8×8=64 个位置
- 只有 6 个位置是 1（3 对配对×2 因对称）
- 比例：6/64 ≈ **9.4%**（实际数据更稀疏，约 0.5%）

如果不加权，模型直接预测全 0 就能获得 90%+ 的准确率。所以需要 **pos_weight**。

#### 4.2 Adaptive pos_weight

```python
# 计算 per-sample 配对密度
gt_pairs = 3  # 这条 RNA 有 3 对配对
L_eff = 8
pair_per_base = gt_pairs / L_eff = 0.375

# 自适应权重 (v5 配置)
alpha = (pair_per_base / 0.5).clamp(0, 1) = (0.375 / 0.5).clamp(0,1) = 0.75
pos_weight = 20 + 0.75 × (199 - 20) = 20 + 134.25 = 154.25
```

**含义**：正样本（配对=1）的 loss 被放大 **154 倍**，强制模型关注稀少的配对位置。

| 配对密度 | pair_per_base | pos_weight | 策略 |
|---------|:---:|:---:|------|
| 很稀疏 (长非编码 RNA) | 0.05 | ~29 | 谨慎，少预测 |
| 中等密度 (tRNA) | 0.25 | ~109 | 平衡 |
| 高密度 (小 RNA) | 0.50 | ~199 | 积极，多预测 |

#### 4.3 具体 Loss 计算（位置 (3,6)）

这是模型"有点不确定"的位置：
```
GT: x₁[3,6] = 1（确实配对）
模型: logit = +2.1, P = 0.891
```

```python
# Positive sample loss (x₁=1)
bce[3,6] = -pos_weight × log(σ(logit))
         = -154.25 × log(0.891)
         = -154.25 × (-0.1155)
         = 17.83
```

对比一个"确信正确"的位置 (1,8)：
```
GT: x₁[1,8] = 1
模型: logit = +4.2, P = 0.985
bce[1,8] = -154.25 × log(0.985) = -154.25 × (-0.0151) = 2.33
```

**→ 不确定的位置 loss 是确信位置的 7.6 倍**，推动模型提高确定性。

#### 4.4 Focal Loss 调制

```python
# 位置 (3,6): p_correct = 0.891 (正确但不够自信)
focal_weight = (1 - 0.891)^1.5 = 0.109^1.5 = 0.036

# 位置 (1,8): p_correct = 0.985 (非常自信)
focal_weight = (1 - 0.985)^1.5 = 0.015^1.5 = 0.0018

# 位置 (4,5): x₁=0, p_correct = 1-0.002 = 0.998
focal_weight = (1 - 0.998)^1.5 = 0.002^1.5 = 0.000089
```

**效果**：
- 已经很自信的位置 → focal_weight ≈ 0 → 几乎不贡献 loss
- 不确定的位置 → focal_weight 大 → 是主要的训练信号来源

```python
final_bce[3,6] = bce[3,6] × focal_weight = 17.83 × 0.036 = 0.642
final_bce[1,8] = bce[1,8] × focal_weight = 2.33 × 0.0018 = 0.0042
```

#### 4.5 Time Weighting

```python
# t=0.6 时的时间权重
w(t) = 1 / (1 - t × (1 - ρ₀))
     = 1 / (1 - 0.6 × 0.995)
     = 1 / (1 - 0.597)
     = 1 / 0.403
     = 2.48
```

| t | w(t) | 含义 |
|:-:|:----:|------|
| 0.0 | 1.00 | 全噪声，loss 权重低 |
| 0.3 | 1.43 | 少量信号 |
| 0.5 | 2.01 | 一半信号 |
| 0.7 | 3.32 | 大量信号 |
| 0.9 | 9.95 | 接近 GT，要求精确，权重很高 |
| 0.99 | 99.5 | 极接近 GT，极高权重 |

**直觉**：接近 t=1 时模型看到的几乎就是答案，此时犯错代价更高。

#### 4.6 汇总 Loss

```python
total_bce = mean(final_bce × time_weight × valid_mask)
# valid_mask: |i-j| >= 3 的位置 (太近的碱基物理上不可能配对)

# 加上辅助损失
stacking_loss = -0.05 × avg(logit[i,j] × logit[i+1,j-1])  # 鼓励 stem 连续
nc_loss = 0.02 × sum(relu(row_sum(sigmoid(logit)) - 1))    # 惩罚一行多个1
density_loss = 0.2 × MSE(density_pred, gt_density)          # 密度回归

total_loss = total_bce + stacking_loss + nc_loss + density_loss
```

---

## 四、推理过程：20 步 τ-leap 采样

### 概述

```
t:  0.00 → 0.05 → 0.10 → ... → 0.95 → 1.00
     ↑                                    ↑
    纯噪声                            生成结果
    
x:  几乎全0 ────→ 逐步"翻转" bits ────→ 接近 GT 的结构
```

### Step 0: 初始化

```python
# 从 prior 采样（极稀疏的随机 1）
x_0 = Bernoulli(ρ₀=0.005)  # 每个位置只有 0.5% 概率为 1
x_0 = symmetrize(x_0)      # 保持对称

# L=8: 64 个位置，期望只有 64×0.005=0.32 个 1，大概率全 0
```

假设初始 x₀ 全为 0（最大概率情况）：
```
x₀:
    1 2 3 4 5 6 7 8
  [ 0 0 0 0 0 0 0 0 ]
  [ 0 0 0 0 0 0 0 0 ]
  ...全0...
```

### 时间步调度 (Cosine Schedule)

```python
# 20 步的时间划分（非均匀！）
dt = [sin(π(k+0.5)/40) / Σ for k=0..19]

# 实际的 dt 值（近似）:
step 0:  dt=0.0039  t: 0.000 → 0.004   (极小步，热身)
step 1:  dt=0.0118  t: 0.004 → 0.016
step 2:  dt=0.0196  t: 0.016 → 0.035
...
step 9:  dt=0.0707  t: 0.302 → 0.373   (中间大步)
step 10: dt=0.0759  t: 0.373 → 0.449
...
step 17: dt=0.0466  t: 0.882 → 0.929
step 18: dt=0.0275  t: 0.929 → 0.956
step 19: dt=0.0118  t: 0.956 → 0.968   (末尾小步，精修)
```

**设计思路**：中间步最大（快速建立结构），首尾步小（避免剧烈变化）。

### Step 1 详细计算（以 step 5 为例）

假设此时 t=0.15, dt=0.04, 当前 x_t 已经有了一些 1：

```
x_t (t=0.15):
    1 2 3 4 5 6 7 8
1 [ 0 0 0 0 0 0 0 1 ]  ← 之前某步翻了 (1,8)
2 [ 0 0 0 0 0 0 0 0 ]
3 [ 0 0 0 0 0 0 0 0 ]
4 [ 0 0 0 0 0 0 0 0 ]
5 [ 0 0 0 0 0 0 0 0 ]
6 [ 0 0 0 0 0 0 0 0 ]
7 [ 0 0 0 0 0 0 0 0 ]
8 [ 1 0 0 0 0 0 0 0 ]
```

#### 1. 模型前向

```python
logit = model(x_t, t=0.15, conditions)
p_x1 = sigmoid(logit)
```

假设模型输出的 p_x1（关键位置）：
```
p_x1[1,8] = 0.92   ← 已经翻了，模型确认是对的
p_x1[2,7] = 0.78   ← 还没翻，但模型很想翻
p_x1[3,6] = 0.65   ← 有倾向但不太确定
p_x1[5,6] = 0.03   ← 确定不该配对
p_x1[4,5] = 0.001  ← 几乎确定不配对
```

#### 2. 计算 CTMC 翻转率

```python
# 对于 0→1 的跳转 (想把 0 变成 1)
rate_01[i,j] = max(p_x1[i,j] - ρ₀, 0) / P(x_t[i,j]=0)

# P(x_t=0) 的边际概率
P(x_t[i,j]=0) = 1 - ((1-t)*ρ₀ + t*p_x1[i,j])
```

**位置 (2,7)：当前 x_t=0，想翻成 1**
```
P(x_t=0) = 1 - (0.85×0.005 + 0.15×0.78) = 1 - (0.00425 + 0.117) = 0.879
rate_01[2,7] = max(0.78 - 0.005, 0) / 0.879 = 0.775 / 0.879 = 0.881
```

**位置 (4,5)：当前 x_t=0，不想翻**
```
P(x_t=0) = 1 - (0.85×0.005 + 0.15×0.001) = 1 - 0.00440 = 0.9956
rate_01[4,5] = max(0.001 - 0.005, 0) / 0.9956 = 0 / 0.9956 = 0
```

**位置 (1,8)：当前 x_t=1，想保持**
```
P(x_t=1) = (1-t)*ρ₀ + t*p_x1 = 0.85×0.005 + 0.15×0.92 = 0.14225
rate_10[1,8] = max(0.005 - 0.92, 0) / 0.14225 = 0 / 0.14225 = 0
```
→ 不会翻回 0！

#### 3. τ-leap 翻转

```python
# 翻转概率 = rate × dt
flip_prob_01[2,7] = rate_01[2,7] × dt = 0.881 × 0.04 = 0.0352
flip_prob_01[4,5] = 0 × 0.04 = 0

# 采样 (伯努利)
rand[2,7] = 0.02  < 0.0352  → ✅ 翻转！ (2,7) 从 0 变 1
```

#### 4. Density-guided damping (v5 特有)

```python
# 如果 density_pred 很低（稀疏 RNA），减少 0→1 的翻转
damp = (2.0 × density_pred).clamp(max=1.0)

# 假设 density_pred = 0.15 (较低)
damp = (2.0 × 0.15).clamp(max=1.0) = 0.30

# 调整后:
flip_prob_01[2,7] = 0.0352 × 0.30 = 0.0106   ← 更保守！
```

**含义**：对于低密度 RNA，模型更加"克制"，避免过预测。

#### 5. 更新 x_t

```python
# 假设这步 (2,7) 确实被翻了
x_t[2,7] = 1
x_t[7,2] = 1  # 对称！
```

---

### 完整 20 步采样示例

| Step | t | dt | 翻转事件 | x_t 中 1 的个数 |
|:----:|:--:|:---:|---------|:--------------:|
| 0 | 0.000 | 0.004 | 无 | 0 |
| 1 | 0.004 | 0.012 | 无 | 0 |
| 2 | 0.016 | 0.020 | (1,8)→1 | 2 |
| 3 | 0.035 | 0.027 | 无 | 2 |
| 4 | 0.063 | 0.035 | (2,7)→1 | 4 |
| 5 | 0.098 | 0.041 | 无 | 4 |
| 6 | 0.139 | 0.047 | (3,6)→1 | 6 |
| 7 | 0.186 | 0.053 | 无 | 6 |
| 8 | 0.240 | 0.059 | 无 | 6 |
| 9 | 0.299 | 0.064 | 无 | 6 |
| 10| 0.363 | 0.068 | (5,6)→1 噪声翻入 | 8 |
| 11| 0.431 | 0.071 | 无 | 8 |
| 12| 0.502 | 0.073 | 无 | 8 |
| 13| 0.575 | 0.074 | (5,6)→0 噪声被纠正 | 6 |
| 14| 0.649 | 0.073 | 无 | 6 |
| 15| 0.722 | 0.070 | 无 | 6 |
| 16| 0.792 | 0.065 | 无 | 6 |
| 17| 0.857 | 0.059 | 无 | 6 |
| 18| 0.916 | 0.050 | 无 | 6 |
| 19| 0.966 | 0.034 | 无 | 6 |

**观察**：
1. **结构逐渐出现**：配对 (1,8) 最先出现（模型最确信），然后是 (2,7)、(3,6)
2. **噪声被纠正**：step 10 不小心翻入了错误的 (5,6)，到 step 13 被 rate_10 翻回
3. **后期稳定**：t>0.7 后几乎不再变化

---

## 五、最终投影 (Greedy Max-Matching)

采样结束后，x_t 可能仍不满足 RNA 物理约束。例如：

```
x_t 的最终 sigmoid(logit):
  (1,8): 0.99   ← 确定配对
  (2,7): 0.97   ← 确定配对
  (3,6): 0.91   ← 确定配对
  (3,5): 0.12   ← 噪声，碱基 3 不能同时配 5 和 6
```

**Greedy Projection 算法**：

```python
def project(scores, L):
    """保证: 对称 + 每行至多1个1 + |i-j|>=3"""
    result = zeros(L, L)
    
    # 1. 清除 |i-j| < 3 的位置
    scores[|i-j| < 3] = 0
    
    # 2. 按 score 从高到低贪心选配对
    while True:
        (i, j) = argmax(scores)
        if scores[i,j] <= threshold:
            break
        
        # 选定 (i,j) 配对
        result[i,j] = result[j,i] = 1
        
        # 删除 i 和 j 的所有其他候选
        scores[i, :] = 0   # 碱基 i 已配对，清除整行
        scores[:, i] = 0
        scores[j, :] = 0   # 碱基 j 已配对，清除整行
        scores[:, j] = 0
    
    return result
```

**执行过程**：
```
第 1 轮: 选 (1,8), score=0.99 → result[(1,8)]=1, 清除 row 1,8
第 2 轮: 选 (2,7), score=0.97 → result[(2,7)]=1, 清除 row 2,7
第 3 轮: 选 (3,6), score=0.91 → result[(3,6)]=1, 清除 row 3,6
第 4 轮: 选 (3,5), score=0.12 → 但 row 3 已清除！跳过
第 5 轮: 没有有效候选了 → 停止
```

**最终输出**：
```
result = {(1,8), (2,7), (3,6)} + 对称  → 完美匹配 GT！
```

---

## 六、数学推导：为什么这套公式能 work？

### 6.1 Flow Matching 的理论基础

**目标**：学习一个"概率路径"，从 prior π₀ 流向数据分布 π₁。

对于离散空间（每个位置 ∈ {0, 1}），定义边际概率路径：

$$p_t(x_{ij} = 1 \mid x_1) = (1-t) \cdot \rho_0 + t \cdot x_1[i,j]$$

**这个公式的含义**：
- **线性插值**：从 $\rho_0$（prior 中"1"的概率）线性走向 $x_1[i,j]$（GT 值）
- **条件路径**：给定目标 $x_1$，描述从 noise 到 signal 的确定性轨迹

### 6.2 CTMC Rate 的推导

连续时间马尔可夫链 (CTMC) 的跳转率定义为：

$$R_{0 \to 1}(i,j,t) = \frac{d}{dt} \frac{p_t(x=1)}{p_t(x=0)}$$

具体地：
$$R_{0 \to 1} = \frac{\max(p_{x_1} - \rho_0, 0)}{p_t(x_{ij}=0)}$$

其中 $p_t(x_{ij}=0) = 1 - p_t(x_{ij}=1)$。

**直觉**：
- 分子 = "模型觉得应该是 1 的程度" 减去 "先验觉得应该是 1 的程度"
- 分母 = "当前这个位置确实是 0 的概率"
- 合起来 = 条件概率：**在已知 x_t=0 的情况下，应该以多大的速率翻成 1**

### 6.3 时间权重的来源

训练 loss 的 importance weight：

$$w(t) = \frac{1}{1 - t(1 - \rho_0)}$$

这来自 ELBO 推导中的 divergence 项。直觉是：
- t→0: marginal 几乎是 prior，模型预测差别不大，权重低
- t→1: marginal 接近 GT，此时模型必须精确，权重高

---

## 七、与其他方法的对比

### vs. 直接预测 (UFold, SPOT-RNA 等)

```
直接预测:   sequence → model → sigmoid → threshold → contact map
SymFold:    sequence → model × 20步 → 逐步翻转 → projection → contact map
```

| 方面 | 直接预测 | SymFold (Flow Matching) |
|------|---------|------------------------|
| 全局一致性 | ❌ 每个位置独立预测 | ✅ 迭代精修，后面步看前面结果 |
| 物理约束 | ❌ 需后处理 | ✅ 可在采样中嵌入约束 |
| 不确定性 | ❌ 单一输出 | ✅ 多次采样得到多个候选 |
| 速度 | ✅ 单次前向 | ❌ 20次前向 |
| 长程依赖 | 取决于感受野 | ✅ 逐步建立，由粗到细 |

### vs. 连续扩散 (RNADiffFold)

RNADiffFold 用的是连续高斯扩散：
```python
# RNADiffFold: 连续
x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise   # x ∈ ℝ
# 最后需要: round(sigmoid(x)) → {0,1}

# SymFold: 离散
x_t ~ Bernoulli(t*x_1 + (1-t)*rho_0)                  # x ∈ {0,1}
# 天然二值，不需要 rounding
```

**SymFold 优势**：
1. 不存在"连续→离散"的 gap
2. 采样过程中可以施加硬约束（对称化）
3. 参数量只有 RNADiffFold 的 1/5

---

## 八、代码对照

### 核心文件

| 概念 | 文件 | 关键函数 |
|------|------|---------|
| 构造 x_t | `src/v4/discrete_flow.py` | `sample_x_t_given_x_1()` |
| 计算 Loss | `src/v4/discrete_flow.py` | `BernoulliFlowLoss_v4.forward()` |
| CTMC Rate | `src/v5/model.py` | `compute_ctmc_rates()` (内联在 `sample()` 中) |
| τ-leap 采样 | `src/v5/model.py` | `sample()` 方法 |
| Greedy Projection | `src/v4/discrete_flow.py` | `project_to_valid_contact_map()` |
| Density-guided | `src/v5/model.py` | `sample()` 中的 `damp` 计算 |

### 训练核心代码 (简化)

```python
# train/train_v5.py 核心循环

for batch in dataloader:
    # 1. 准备数据
    contact = batch['contact_map']       # GT: (B, 1, L, L)
    tokens = batch['tokens']             # RNA-FM input
    data_fcn_2 = batch['fcn_features']   # 17ch for UFold
    
    # 2. 模型前向 (内部完成: 采样 t, 构造 x_t, 计算 logit)
    logit, density_pred, loss_dict = model(
        contact, data_fcn_2, tokens, contact_masks, seq_oh
    )
    # model.forward() 内部:
    #   t = rand(B)
    #   x_t = sample_x_t_given_x_1(x_1, t, rho_0)
    #   logit = backbone(x_t, t, conditions)
    #   loss = BernoulliFlowLoss_v4(logit, x_1, t, contact_masks)
    
    # 3. 反向传播
    loss = loss_dict['total']
    loss.backward()
    optimizer.step()
```

### 推理核心代码 (简化)

```python
# model.sample() 方法

@torch.no_grad()
def sample(self, data_fcn_2, tokens, contact_masks, seq_oh):
    # 1. 初始化
    x_t = Bernoulli(rho_0).sample()       # (B, 1, L, L)
    x_t = symmetrize(x_t) * contact_masks
    
    # 2. 预测 density (v5)
    _, density_pred = self.backbone(x_t, t=0.5, ..., return_density=True)
    damp = (2.0 * density_pred).clamp(max=1.0)
    
    # 3. 20 步 τ-leap
    for k in range(20):
        t = sum(dt_list[:k])
        dt = dt_list[k]
        
        logit = self.backbone(x_t, t, conditions)
        p_x1 = sigmoid(logit)
        
        # CTMC rates
        rate_01 = max(p_x1 - rho_0, 0) / P(x_t=0)
        rate_10 = max(rho_0 - p_x1, 0) / P(x_t=1)
        
        rate_01 = rate_01 * damp                    # v5: density damping
        
        # τ-leap flip
        flip_01 = Bernoulli(rate_01 * dt) & (x_t == 0)
        flip_10 = Bernoulli(rate_10 * dt) & (x_t == 1)
        x_t = x_t + flip_01 - flip_10
        x_t = symmetrize(x_t) * contact_masks
    
    # 4. Greedy projection
    result = greedy_max_matching(x_t, sigmoid(logit))
    return result
```

---

## 九、总结

Discrete Flow Matching 的核心是三个优雅的设计：

1. **前向插值** — 用 Bernoulli 概率在 {先验噪声 ρ₀} 和 {GT x₁} 之间线性插值
2. **神经网络去噪** — 模型看着"部分暴露"的 x_t，预测完整的 P(x₁=1)
3. **τ-leap 采样** — 用 CTMC 理论计算"每个 bit 应该以多大概率翻转"

```
                    训练
    ┌────────────────────────────────────────┐
    │  x₁ (GT)                              │
    │    ↓  Bernoulli(t·x₁ + (1-t)·ρ₀)     │
    │  x_t (噪声版)                          │
    │    ↓  model(x_t, t)                    │
    │  logit → Loss(logit, x₁)              │
    └────────────────────────────────────────┘

                    推理
    ┌────────────────────────────────────────┐
    │  x₀ ~ Bernoulli(ρ₀)  (随机初始化)      │
    │    ↓                                   │
    │  for k = 1..20:                        │
    │    p_x1 = sigmoid(model(x_t, t))       │
    │    rates = CTMC(p_x1, x_t, t, ρ₀)     │
    │    x_t += τ-leap flips                 │
    │    ↓                                   │
    │  greedy_projection(x_t, scores)        │
    │    ↓                                   │
    │  ✓ 合法 RNA 接触图                      │
    └────────────────────────────────────────┘
```

# V3 架构设计 — RNA-Aware DiT + Discrete Flow Matching

> 从 RNA 二级结构预测的物理本质出发，重新设计 DiT 架构与 Flow Matching 框架。

---

## 一、RNA 二级结构的本质特性

在设计模型之前，必须深刻理解我们要预测什么：

### 1.1 配对矩阵的结构性质

```
RNA 序列:  A U G C G C A U
配对关系:  (1,8) (2,7) (3,6)  → 形成一个 stem

配对矩阵 C[i,j]:
    A U G C G C A U
A [ 0 0 0 0 0 0 0 1 ]
U [ 0 0 0 0 0 0 1 0 ]
G [ 0 0 0 0 0 1 0 0 ]
C [ 0 0 0 0 0 0 0 0 ]   ← unpaired
G [ 0 0 0 0 0 0 0 0 ]   ← unpaired
C [ 0 0 1 0 0 0 0 0 ]
A [ 0 1 0 0 0 0 0 0 ]
U [ 1 0 0 0 0 0 0 0 ]
```

### 1.2 关键观察

| 性质 | 描述 | 对模型的启示 |
|------|------|------------|
| **极度稀疏** | L=100 时约 30 对，ρ ≈ 0.6% | Loss 必须处理 1:170 的不平衡 |
| **对称** | C[i,j] = C[j,i] | 模型必须结构性对称 |
| **每行至多1** | 每个碱基最多配一个 partner | 这是**组合约束**，不是独立 per-pixel 问题 |
| **Stem 结构** | 配对成堆出现：(i,j),(i+1,j-1),(i+2,j-2)... | 配对矩阵中呈**反对角线**分布 |
| **嵌套性** | canonical 结构不允许交叉配对 | 约束了配对的全局拓扑 |
| **碱基互补** | 只有 AU/GC/GU 能配对 | 输入序列直接限定了可能配对位置 |
| **长程依赖** | 远距离碱基可以配对 (e.g. i=10, j=200) | 需要全局 attention |

### 1.3 核心洞察：这不是 "图像去噪"

当前做法将配对矩阵当作二值图像，用 2D DiT 去噪。但 RNA 配对矩阵 **不是图像**：

- 图像：像素间有局部相关性、连续值、无硬组合约束
- 配对矩阵：**组合结构**（每行至多一个 1）、**反对角线模式**（stem）、**嵌套拓扑**

**关键问题**: 现有的 per-element BCE loss 把每个 C[i,j] 当作独立 Bernoulli 变量来训练。但实际上 C[i,j]=1 意味着 C[i,k]=0 ∀k≠j（互斥约束）。这个约束在训练中完全被忽略了，只在推理最后靠 projection 强制满足。

---

## 二、设计原则

基于上述分析，v3 架构的核心设计原则：

1. **约束内置**: 每行至多一配对的约束必须融入模型本身，而非后处理
2. **序列感知**: RNA 是 1D 序列折叠成 2D 结构，模型应尊重这一层次
3. **Stem 感知**: 配对成堆出现（反对角线），模型应有对应的归纳偏置
4. **Flow Matching 保留**: 离散 flow matching 是好框架（v1 已证明），但需要更好的参数化
5. **简洁稳定**: 避免 v2 的过度设计（U-Net 下采样等），保持训练稳定性

---

## 三、V3 架构：Pair-Aware DiT (PA-DiT)

### 3.1 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                    PA-DiT 架构总览                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: x_t ∈ {0,1}^{L×L} (当前状态) + t (时间)         │
│         + sequence S (RNA序列)                           │
│                                                         │
│  ┌─────────────────────────────────────────┐            │
│  │  Sequence Encoder (1D Transformer)       │            │
│  │  S → h_seq ∈ R^{L×D_seq}               │            │
│  │  (RNA-FM frozen + 轻量 adapter)          │            │
│  └──────────────────┬──────────────────────┘            │
│                     │                                    │
│                     ▼                                    │
│  ┌─────────────────────────────────────────┐            │
│  │  Pair Representation Builder             │            │
│  │  h_seq → z_pair ∈ R^{L×L×D_pair}       │            │
│  │  (outer sum + bias + triangular update)  │            │
│  └──────────────────┬──────────────────────┘            │
│                     │                                    │
│                     ▼                                    │
│  ┌─────────────────────────────────────────┐            │
│  │  Denoising Backbone (Axial DiT, N layers)│            │
│  │  (x_t_embed, z_pair, t) → logit         │            │
│  │  每层: RowAttn + ColAttn + TriangleUpdate │            │
│  └──────────────────┬──────────────────────┘            │
│                     │                                    │
│                     ▼                                    │
│  ┌─────────────────────────────────────────┐            │
│  │  Row-Softmax Output Head                 │            │
│  │  logit → P(partner_i = j)  per-row      │            │
│  │  (每行一个分类分布，天然满足至多1配对)     │            │
│  └─────────────────────────────────────────┘            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 核心创新点

#### 创新 1: Row-Softmax 参数化 (替代 per-element Sigmoid)

**现有方法 (v1/v2)**: 输出 L×L logit → 每个元素独立 sigmoid → P(C[i,j]=1)

**问题**: 忽略了 "每行至多一个 1" 的约束，训练和推理不一致。

**V3 方法**: 每行输出一个 **(L+1)-way 分类分布**

```python
# 对位置 i，预测它和谁配对（或不配对）
# logit[i] ∈ R^{L+1}，最后一维是 "unpaired" 类
# P(partner_i = j) = softmax(logit[i])[j]
# P(i is unpaired) = softmax(logit[i])[L]  (最后一个 class)

class RowSoftmaxHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.to_logit = nn.Linear(hidden_dim, 1)  # per-pair score
        self.unpaired_bias = nn.Parameter(torch.zeros(1))  # learnable unpaired baseline
    
    def forward(self, pair_repr):
        """
        pair_repr: (B, L, L, D) — 每对 (i,j) 的表示
        返回: (B, L, L+1) — 每行的配对概率分布
        """
        B, L, _, D = pair_repr.shape
        
        # 每对 (i,j) 的 score
        scores = self.to_logit(pair_repr).squeeze(-1)  # (B, L, L)
        
        # 加入 "unpaired" 选项
        unpaired = self.unpaired_bias.expand(B, L, 1)  # (B, L, 1)
        logits = torch.cat([scores, unpaired], dim=-1)  # (B, L, L+1)
        
        # Mask: |i-j|<4 的位置设为 -inf，padding 设为 -inf
        logits = logits.masked_fill(invalid_mask, float('-inf'))
        
        # Row-softmax: 每行一个分类分布
        probs = F.softmax(logits, dim=-1)  # (B, L, L+1)
        
        return probs[:, :, :L]  # (B, L, L) 配对概率
```

**优势**:
- ✅ **约束内置**: softmax 天然保证每行概率和 ≤ 1
- ✅ **训练-推理一致**: 训练时就在学一个"选择 partner"的分布
- ✅ **无需 pos_weight**: 不再是严重不平衡的二分类，而是多分类
- ✅ **投影简化**: 推理时只需取每行 argmax + 对称化协商

**对称化**: 最终配对矩阵需要 i→j 和 j→i 都同意：
```python
def symmetrize_probs(probs):
    """P_final(i,j) = P(i→j) * P(j→i) 的归一化"""
    joint = probs * probs.transpose(-2, -1)  # 双向一致性
    return joint / joint.sum(dim=-1, keepdim=True).clamp(min=1e-8)
```

---

#### 创新 2: Triangle Update (受 AlphaFold2 启发)

RNA 配对矩阵有**传递性**：如果 (i,j) 配对且 (i,k) 不配对，那么 k 和 j 的关系也受约束。AlphaFold2 的 Triangle Multiplicative Update 正好捕获这种三体关系。

```python
class TriangleMultiplication(nn.Module):
    """
    三角更新: z[i,j] 受 z[i,k] 和 z[k,j] 影响 (对所有 k)
    这自然编码了 "每行至多一对" 的竞争关系
    """
    def __init__(self, dim, mode='outgoing'):
        super().__init__()
        self.mode = mode  # 'outgoing' or 'incoming'
        self.proj_left = nn.Linear(dim, dim)
        self.proj_right = nn.Linear(dim, dim)
        self.gate_left = nn.Linear(dim, dim)
        self.gate_right = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, z):
        """z: (B, L, L, D)"""
        z_norm = self.layer_norm(z)
        
        left = torch.sigmoid(self.gate_left(z_norm)) * self.proj_left(z_norm)
        right = torch.sigmoid(self.gate_right(z_norm)) * self.proj_right(z_norm)
        
        if self.mode == 'outgoing':
            # z[i,j] += Σ_k left[i,k] * right[j,k]
            update = torch.einsum('bikd,bjkd->bijd', left, right)
        else:  # incoming
            # z[i,j] += Σ_k left[k,i] * right[k,j]
            update = torch.einsum('bkid,bkjd->bijd', left, right)
        
        return self.out_proj(update)
```

**为什么这对 RNA 特别有效**:
- `z[i,k]` 和 `z[j,k]` 的乘积捕获 "i 和 j 通过 k 的关系"
- 当 i 已经和某个 k 配对（z[i,k] 大），triangle update 会自动抑制 z[i,j]（竞争效应）
- 这正是 "每行至多一配对" 的软版本！

---

#### 创新 3: Anti-Diagonal Convolution (Stem 感知)

RNA 的 stem 结构表现为配对矩阵中的**反对角线条带**：
```
(i,j), (i+1,j-1), (i+2,j-2), ...
```

标准 2D Conv 或 Axial Attention 对这种反对角线模式没有特殊偏置。我们设计一个专门的模块：

```python
class AntiDiagonalConv(nn.Module):
    """
    沿反对角线方向的 1D 卷积
    捕获 stem 的连续配对模式
    """
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, z):
        """z: (B, L, L, D)"""
        B, L, _, D = z.shape
        
        # 提取所有反对角线 (i+j=const → 同一条反对角线)
        # 反对角线 d: {(i, d-i) | max(0,d-L+1) <= i <= min(d, L-1)}
        # 共 2L-1 条反对角线
        
        z_norm = self.norm(z)
        output = torch.zeros_like(z)
        
        for d in range(2*L - 1):
            i_start = max(0, d - L + 1)
            i_end = min(d, L - 1)
            length = i_end - i_start + 1
            if length < 2:
                continue
            
            # 提取该反对角线上的 token
            indices_i = torch.arange(i_start, i_end + 1)
            indices_j = d - indices_i
            diag_tokens = z_norm[:, indices_i, indices_j, :]  # (B, length, D)
            
            # 1D 卷积
            out = self.conv(diag_tokens.transpose(1, 2)).transpose(1, 2)  # (B, length, D)
            output[:, indices_i, indices_j, :] = out
        
        # Gated residual
        gate = torch.sigmoid(self.gate(z))
        return gate * output
```

**注意**: 上面的纯 Python 循环效率低。实际实现可用 `torch.diagonal` + padding 做高效批量处理：

```python
def forward_efficient(self, z):
    """高效实现: 利用 torch.diagonal 批量处理所有反对角线"""
    B, L, _, D = z.shape
    z_norm = self.norm(z)
    
    # 翻转 j 轴使反对角线变成主对角线: z_flip[i, L-1-j] = z[i, j]
    z_flip = z_norm.flip(dims=[2])  # (B, L, L, D)
    
    # 现在 stem 模式变成了主对角线方向 → 可用对角线卷积
    # reshape 为 (B*D, L, L) 后用 depthwise conv2d kernel=(k,1) 沿对角?
    # 或者更直接: 用 unfold 沿对角方向取 window
    
    # 实际可用: 对每个对角偏移 d，提取 band，做 1D conv
    # 利用 F.pad + 循环移位 实现 O(L*kernel) 复杂度
    ...
```

**简化替代方案**：如果反对角线卷积实现复杂，可以用一个旋转 45° 的 depthwise conv2d：
```python
# 旋转坐标系: (i,j) → (i+j, i-j) 使反对角线变为水平线
# 然后沿水平方向做 1D conv
```

---

#### 创新 4: Discrete Flow Matching 改进 — Pair-Level 参数化

**现有方法**: 网络预测每个元素的 P(x₁[i,j]=1)，然后用 CTMC rate 做 τ-leap

**问题**: 元素级预测 + 元素级采样，完全忽略了行约束

**V3 方法**: 在 flow matching 框架中引入 **row-level 转移**

```python
def sample_v3(network_fn, L, num_steps=20, rho_0=0.005):
    """
    V3 采样: Row-categorical τ-leap
    
    状态空间: 每行 i 的状态是 partner(i) ∈ {0,1,...,L-1, unpaired}
    而非每个 (i,j) 的 0/1
    """
    # 初始化: 每行独立从 categorical 先验采样
    # 先验: P(partner_i = j) = rho_0 / L (配对任意位置的均匀小概率)
    #        P(unpaired) = 1 - rho_0 (大概率不配对)
    partner = torch.full((B, L), L, dtype=torch.long)  # 初始全 unpaired
    
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = k * dt
        
        # 构建当前配对矩阵 x_t
        x_t = partner_to_matrix(partner, L)  # (B, L, L)
        
        # 网络预测 row-softmax 分布
        probs = network_fn(x_t, t)  # (B, L, L+1), probs[b,i,:] = P(partner_i=·|x_t, t)
        
        # Row-level CTMC: 每行独立做 categorical 跳转
        for i in range(L):
            current = partner[:, i]  # 当前 partner
            # 跳转率: 从 current 跳到其他状态
            target_prob = probs[:, i, :]  # (B, L+1)
            rate = compute_row_rate(current, target_prob, t, rho_0)
            
            # τ-leap: 以 rate*dt 概率跳转到新状态
            if random < rate * dt:
                new_partner = sample_from(target_prob)
                partner[:, i] = new_partner
        
        # 对称化协商: 如果 partner[i]=j 但 partner[j]≠i，需要冲突解决
        partner = symmetrize_partners(partner, probs)
    
    return partner_to_matrix(partner, L)
```

**对称化协商策略**:
```python
def symmetrize_partners(partner, probs):
    """
    当 partner[i]=j 但 partner[j]≠i 时，用双方概率解决冲突
    保留 P(i→j) * P(j→i) 更大的配对
    """
    B, L = partner.shape
    for b in range(B):
        for i in range(L):
            j = partner[b, i]
            if j < L and partner[b, j] != i:
                # 冲突: i 想配 j，但 j 不想配 i
                score_ij = probs[b, i, j] * probs[b, j, i]
                score_jk = probs[b, j, partner[b,j]] * probs[b, partner[b,j], j] \
                           if partner[b,j] < L else 0
                if score_ij > score_jk:
                    # i-j 配对赢
                    old_k = partner[b, j]
                    partner[b, j] = i
                    if old_k < L:
                        partner[b, old_k] = L  # old_k 变 unpaired
                else:
                    partner[b, i] = L  # i 变 unpaired
    return partner
```

---

### 3.3 完整 Block 设计

```python
class PADiTBlock(nn.Module):
    """
    Pair-Aware DiT Block
    
    每个 block 包含:
    1. Row Axial Attention (行内 pair 竞争)
    2. Column Axial Attention (列内 pair 竞争)  
    3. Triangle Multiplicative Update (三体关系)
    4. Anti-Diagonal Conv (stem 模式, 可选)
    5. FFN
    
    所有子层都用 AdaLN-Zero 调制 (时间 + 全局条件)
    """
    def __init__(self, dim, num_heads, use_triangle=True, use_antidiag=False):
        super().__init__()
        # Sub-layers
        self.row_attn = SharedAxialAttention(dim, num_heads)
        self.triangle_out = TriangleMultiplication(dim, mode='outgoing') if use_triangle else None
        self.triangle_in = TriangleMultiplication(dim, mode='incoming') if use_triangle else None
        self.antidiag_conv = AntiDiagonalConv(dim) if use_antidiag else None
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
        
        # AdaLN-Zero for each sub-layer
        num_sublayers = 3 + int(use_triangle) * 2 + int(use_antidiag)
        self.adaLN = nn.Linear(dim, num_sublayers * 3 * dim)  # shift, scale, gate per sublayer
        
        # Layer norms
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_sublayers)])
    
    def forward(self, z, cond):
        """
        z: (B, L, L, D) — pair representation
        cond: (B, D) — global condition (t + seq info)
        """
        params = self.adaLN(cond).chunk(num_sublayers * 3, dim=-1)
        idx = 0
        
        # 1. Axial Attention (行 + 列)
        z = z + self._gated_sublayer(self.row_attn, z, self.norms[idx], params[idx*3:(idx+1)*3])
        idx += 1
        
        # 2. Triangle Update (outgoing)
        if self.triangle_out is not None:
            z = z + self._gated_sublayer(self.triangle_out, z, self.norms[idx], params[idx*3:(idx+1)*3])
            idx += 1
        
        # 3. Triangle Update (incoming)
        if self.triangle_in is not None:
            z = z + self._gated_sublayer(self.triangle_in, z, self.norms[idx], params[idx*3:(idx+1)*3])
            idx += 1
        
        # 4. Anti-Diagonal Conv (optional, 后几层启用)
        if self.antidiag_conv is not None:
            z = z + self._gated_sublayer(self.antidiag_conv, z, self.norms[idx], params[idx*3:(idx+1)*3])
            idx += 1
        
        # 5. FFN
        z = z + self._gated_sublayer(self.ffn, z, self.norms[idx], params[idx*3:(idx+1)*3])
        
        return z
    
    def _gated_sublayer(self, fn, x, norm, params):
        shift, scale, gate = params
        h = norm(x) * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
        return gate.unsqueeze(1).unsqueeze(1) * fn(h)
```

---

### 3.4 条件注入策略

```python
class SequenceConditioner(nn.Module):
    """
    从 RNA 序列提取条件信息，注入到 pair representation
    """
    def __init__(self, fm_dim=640, hidden_dim=128, pair_dim=64):
        super().__init__()
        # RNA-FM adapter (frozen backbone + trainable adapter)
        self.fm_adapter = nn.Sequential(
            nn.Linear(fm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Sequence → Pair: outer sum (类似 AlphaFold2 的 pair representation 初始化)
        self.left_proj = nn.Linear(hidden_dim, pair_dim)
        self.right_proj = nn.Linear(hidden_dim, pair_dim)
        
        # 碱基互补偏置: 可学习的 4×4 碱基对 embedding
        self.base_pair_embed = nn.Embedding(16, pair_dim)  # 4*4 种碱基对组合
        
        # 相对位置编码 (1D 序列距离 |i-j|)
        self.rel_pos_embed = nn.Embedding(640, pair_dim)  # max_len 个不同距离
        
        # UFold condition
        self.ufold_proj = nn.Conv2d(8, pair_dim, 1)
    
    def forward(self, fm_emb, seq_idx, ufold_cond):
        """
        fm_emb: (B, L, 640) — RNA-FM output
        seq_idx: (B, L) — 碱基索引 (A=0, U=1, G=2, C=3)
        ufold_cond: (B, 8, L, L) — UFold condition
        
        返回: z_init (B, L, L, pair_dim) — 初始 pair representation
        """
        B, L, _ = fm_emb.shape
        
        # 1. Sequence embedding
        h = self.fm_adapter(fm_emb)  # (B, L, hidden_dim)
        
        # 2. Outer sum → pair
        left = self.left_proj(h)   # (B, L, pair_dim)
        right = self.right_proj(h) # (B, L, pair_dim)
        z_outer = left.unsqueeze(2) + right.unsqueeze(1)  # (B, L, L, pair_dim)
        
        # 3. Base-pair embedding
        bp_idx = seq_idx.unsqueeze(2) * 4 + seq_idx.unsqueeze(1)  # (B, L, L)
        z_bp = self.base_pair_embed(bp_idx)  # (B, L, L, pair_dim)
        
        # 4. Relative position
        pos = torch.arange(L, device=fm_emb.device)
        rel_pos = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs().clamp(max=639)  # (L, L)
        z_pos = self.rel_pos_embed(rel_pos).unsqueeze(0)  # (1, L, L, pair_dim)
        
        # 5. UFold condition
        z_ufold = self.ufold_proj(ufold_cond).permute(0, 2, 3, 1)  # (B, L, L, pair_dim)
        
        # 6. 融合
        z_init = z_outer + z_bp + z_pos + z_ufold
        
        return z_init
```

---

### 3.5 Flow Matching 的 Loss 改进

```python
class PairFlowLoss(nn.Module):
    """
    V3 Loss: Row-Softmax Cross Entropy + Structure Regularization
    
    不再是 per-element BCE，而是 per-row classification loss
    """
    def __init__(self, rho_0=0.005, lambda_sym=0.1, lambda_stem=0.05):
        super().__init__()
        self.rho_0 = rho_0
        self.lambda_sym = lambda_sym
        self.lambda_stem = lambda_stem
    
    def forward(self, row_logits, x_1, t, contact_masks, lengths):
        """
        row_logits: (B, L, L+1) — 网络输出的每行 logit
        x_1: (B, L, L) — ground truth 配对矩阵
        t: (B,) — 时间
        contact_masks: (B, L, L)
        lengths: (B,) — 序列真实长度
        """
        B, L, _ = row_logits.shape
        
        # 1. 构造 per-row target: 每行的 partner index (或 L 表示 unpaired)
        # x_1[i,:] 中唯一的 1 的位置就是 partner, 全 0 则 target = L (unpaired)
        target = x_1.argmax(dim=-1)  # (B, L)
        is_unpaired = (x_1.sum(dim=-1) == 0)  # (B, L)
        target[is_unpaired] = L  # unpaired class
        
        # 2. Mask invalid positions: |i-j| < 4, padding
        row_logits_masked = row_logits.clone()
        for i in range(L):
            row_logits_masked[:, i, max(0,i-3):min(L,i+4)] = float('-inf')  # |i-j|<4
        # padding
        for b in range(B):
            row_logits_masked[b, lengths[b]:, :] = float('-inf')
            row_logits_masked[b, :, lengths[b]:L] = float('-inf')
        
        # 3. Cross Entropy Loss (per-row classification)
        # 注意: 只对有效行 (i < lengths[b]) 计算
        loss_ce = F.cross_entropy(
            row_logits_masked.view(B * L, L + 1),
            target.view(B * L),
            ignore_index=-100,  # padding 行标记为 -100
            reduction='mean'
        )
        
        # 4. 对称一致性 loss: P(i→j) 和 P(j→i) 应该接近
        probs = F.softmax(row_logits_masked, dim=-1)[:, :, :L]  # (B, L, L)
        sym_diff = (probs - probs.transpose(-2, -1)).abs()
        loss_sym = (sym_diff * contact_masks).sum() / contact_masks.sum()
        
        # 5. Stem continuity bonus (可选): 鼓励相邻位置的 partner 也相邻
        # 如果 partner[i] = j, 则鼓励 partner[i+1] = j-1
        loss_stem = self._stem_loss(probs, target, contact_masks)
        
        # 6. 时间加权 (接近 t=1 时 loss 权重更大)
        t_weight = 1.0 / (1.0 - t * (1.0 - self.rho_0))
        loss_ce = loss_ce * t_weight.mean()
        
        return loss_ce + self.lambda_sym * loss_sym + self.lambda_stem * loss_stem
    
    def _stem_loss(self, probs, target, masks):
        """鼓励 stem 连续性: P(partner[i+1] = partner[i]-1)"""
        B, L, _ = probs.shape
        # 对于已知 target[i]=j (j<L), 在 probs[i+1] 上鼓励 j-1 位置的概率高
        # 简化: 计算 probs 沿反对角线方向的平滑度
        # probs[i,j] 和 probs[i+1, j-1] 的差异应该小
        if L < 3:
            return torch.tensor(0.0)
        
        p_shift = probs[:, 1:, :-1]   # probs[i+1, j]  → 对齐到 probs[i, j+1]
        p_orig = probs[:, :-1, 1:]    # probs[i, j+1]
        stem_smooth = (p_shift - p_orig).abs()
        return (stem_smooth * masks[:, 1:, 1:]).mean()
```

---

## 四、完整模型配置

```python
# src/v3/config.py

V3_CONFIG = {
    # Sequence Encoder
    "fm_dim": 640,             # RNA-FM output dim (frozen)
    "seq_hidden_dim": 128,     # FM adapter output
    
    # Pair Representation
    "pair_dim": 64,            # pair representation 维度
    
    # Denoising Backbone
    "backbone_dim": 192,       # DiT hidden dim (与 v1 相同)
    "num_heads": 8,            # attention heads (v1=4 → v3=8)
    "num_layers": 8,           # flat layers (无 U-Net!)
    "use_triangle": True,      # Triangle Update (前 6 层启用)
    "use_antidiag": True,      # Anti-Diagonal Conv (后 4 层启用)
    "patch_size": 1,           # ★ 不再 patch! 直接 per-position 处理
    
    # Output Head
    "output_mode": "row_softmax",  # 'row_softmax' or 'element_sigmoid'
    
    # Flow Matching
    "rho_0": 0.005,
    "num_steps": 20,           # CTMC 采样步数
    
    # Training
    "lr": 5e-5,                # 降低学习率 (v2 用 1e-4 过大)
    "warmup_epochs": 5,
    "batch_size_80": 32,       # 介于 v1(16) 和 v2(128) 之间
    "batch_size_160": 16,
    "batch_size_320": 4,
    "epochs": 200,
    "patience": 30,
    "dropout": 0.15,
    "grad_clip": 1.0,
    
    # Loss
    "lambda_sym": 0.1,
    "lambda_stem": 0.05,
}
```

### 关于 Patch Size 的选择

**v1/v2 用 patch_size=4**: 将 L×L 矩阵降为 (L/4)×(L/4) 的 token grid，每个 token 代表 4×4=16 个元素。

**V3 不用 patch 的原因**:
- Row-Softmax 需要逐行操作，patch 化会破坏行结构
- 配对矩阵太稀疏（每行只有 0 或 1 个 1），4×4 patch 中大概率全 0，信息密度极低
- Triangle Update 需要逐元素的 pair representation

**计算量控制**:
- 不 patch 意味着 token 数 = L×L，L=640 时 token 数 = 409600，太大
- 解决: **Axial Attention** 把复杂度从 O((L²)²) = O(L⁴) 降到 O(L² × L) = O(L³)
- 对于 L=640: 行 attention 每行 640 个 token → 可接受
- Triangle Update: O(L³) 复杂度，L=640 时需要优化（分块/稀疏）

**实际折中方案**: 对长序列 (L>320) 使用 patch_size=2，短序列 patch_size=1
```python
patch_size = 1 if L <= 320 else 2
```

---

## 五、Discrete Flow Matching 适配

### 5.1 前向过程 (不变)

保持 Bernoulli 边际插值，但用 **pair-level** 表示：
```python
# 前向: 对每行的 partner 状态做插值
# t=0: 每行是 "unpaired" (或以 ρ₀ 概率随机配对)
# t=1: 每行是 ground truth partner

def sample_x_t_v3(x_1, t, rho_0=0.005):
    """
    与 v1/v2 完全相同的前向过程 (element-level Bernoulli)
    保持兼容性，变化在网络输出端
    """
    p_one = t.view(-1, 1, 1) * x_1 + (1 - t.view(-1, 1, 1)) * rho_0
    return (torch.rand_like(x_1) < p_one).float()
```

### 5.2 训练 (核心变化在 Loss)

```python
def train_step_v3(model, batch):
    x_1 = batch['contact']  # (B, L, L)
    
    # 1. 随机时间
    t = torch.rand(B)
    
    # 2. 前向加噪
    x_t = sample_x_t_v3(x_1, t)
    x_t = symmetrize(x_t) * mask
    
    # 3. 网络预测 row-softmax logits
    row_logits = model(x_t, t, seq_cond)  # (B, L, L+1)
    
    # 4. Loss: per-row CE + symmetry + stem
    loss = pair_flow_loss(row_logits, x_1, t, mask, lengths)
    
    return loss
```

### 5.3 采样 (核心变化: Row-Categorical CTMC)

```python
@torch.no_grad()
def sample_v3(model, seq_cond, L, num_steps=20, rho_0=0.005):
    """
    Row-Categorical τ-leap 采样
    """
    # 初始化
    x_t = (torch.rand(B, L, L) < rho_0).float()
    x_t = symmetrize(x_t) * mask
    
    dt = 1.0 / num_steps
    
    for k in range(num_steps):
        t = torch.tensor([k * dt])
        
        # 网络预测 row-softmax
        row_logits = model(x_t, t, seq_cond)  # (B, L, L+1)
        probs = F.softmax(row_logits, dim=-1)[:, :, :L]  # (B, L, L)
        
        # 对称化概率
        probs_sym = (probs + probs.transpose(-2, -1)) / 2
        
        # 标准 CTMC τ-leap (element-level, 但 rate 来自 row-softmax)
        p_x1 = probs_sym  # 作为 P(x_1[i,j]=1) 的估计
        rate_01, rate_10 = compute_ctmc_rates(x_t, p_x1, t, rho_0)
        
        flip_01 = (torch.rand_like(x_t) < (rate_01 * dt).clamp(max=1)) & (x_t < 0.5)
        flip_10 = (torch.rand_like(x_t) < (rate_10 * dt).clamp(max=1)) & (x_t > 0.5)
        
        x_t = torch.where(flip_01, torch.ones_like(x_t), x_t)
        x_t = torch.where(flip_10, torch.zeros_like(x_t), x_t)
        x_t = symmetrize(x_t) * mask
    
    # 最终投影: 每行 argmax (因为 row-softmax 天然满足约束)
    final_probs = F.softmax(row_logits, dim=-1)[:, :, :L]
    final_probs_sym = final_probs * final_probs.transpose(-2, -1)
    x_final = greedy_matching(final_probs_sym, mask)  # 仍用贪心确保严格约束
    
    return x_final
```

---

## 六、与 v1/v2 的对比

| 维度 | v1 | v2 | **v3 (proposed)** |
|------|:--:|:--:|:-----------------:|
| **参数化** | element sigmoid | element sigmoid | **row softmax** |
| **Loss** | pos-weighted BCE | pos-weighted BCE | **per-row CE** |
| **backbone** | 6L flat SEDiT | 3+2+3 U-Net MSEDiT | **8L flat PA-DiT** |
| **特有模块** | 无 | local bias | **Triangle + AntiDiag** |
| **约束处理** | 后处理 projection | 后处理 projection | **训练时内置** |
| **Patch size** | 4 | 4 | **1 (或 2)** |
| **类不平衡** | pos_weight=199 | pos_weight=199 | **不需要** (CE 天然平衡) |
| **Stem 感知** | 无 | 无 | **AntiDiag Conv** |
| **三体关系** | 无 | 无 | **Triangle Update** |

---

## 七、计算量估算

### 单层 PADiTBlock 计算量 (L=160, D=192):

| 子模块 | 复杂度 | L=160 FLOPs |
|--------|--------|:-----------:|
| Axial Attn (row+col) | O(L² × L × D) | ~60M |
| Triangle Update × 2 | O(L³ × D) | ~160M |
| AntiDiag Conv | O(L² × D × k) | ~25M |
| FFN | O(L² × D × 4D) | ~57M |
| **单层合计** | | **~300M** |
| **8 层合计** | | **~2.4G** |

对比 v1 (L=160, patch=4 → 40×40 grid):
- 单层: O(40² × 40 × 192) ≈ 12M
- 6 层: ~72M

**V3 计算量约为 v1 的 33 倍**。在 H20 (96GB) 上：
- v1 L=160 batch=8, 每步约 0.1s → v3 约 3.3s
- 但 v3 batch 更小 (16 vs 64)，实际时间差约 8~10x

### 优化策略
1. **长序列用 patch=2**: L>320 时 patch=2，token grid 减半
2. **Triangle Update 稀疏化**: 只对 score > threshold 的 (i,j) 对做 triangle
3. **FlashAttention**: Axial attention 中使用 flash attention
4. **混合精度**: bf16 (H20 支持)

---

## 八、实施路线图

### Phase 1: 最小可行版本 (MVP, 3天)

只实现核心改动，验证 row-softmax 的有效性：

1. 复用 v1 的 flat SEDiT backbone（6 层）
2. 将输出头从 element sigmoid 改为 row softmax
3. Loss 改为 per-row CE
4. 采样保持 element-level CTMC + 最终 greedy matching
5. **不加** Triangle Update 和 AntiDiag（留到 Phase 2）

预期: 如果 row-softmax 有效，应该在 30-50 epoch 内达到 v1 水平 (F1>0.85)

### Phase 2: 加入 Triangle Update (2天)

1. 在每个 block 中加入 TriangleMultiplication (outgoing + incoming)
2. 增加 pair_dim 到 64
3. 加入 base-pair embedding 和 relative position

预期: F1 提升 2-5%

### Phase 3: 加入 Stem 感知 (2天)

1. 实现 AntiDiagonalConv（后 4 层启用）
2. 加入 stem continuity loss
3. 调参

### Phase 4: 去 Patch 化 + 长序列优化 (3天)

1. patch_size=1 (短序列)
2. 长序列策略 (稀疏 triangle, flash attention)
3. 大规模训练 200 epoch

---

## 九、总结

**V3 的核心哲学**: 

> 不要把 RNA 配对矩阵当图像去噪。它是一个**组合匹配问题**——每个碱基选择自己的 partner。模型应该直接学习这个 "选择" 过程。

**三个关键设计决策**:

1. **Row-Softmax 替代 Element-Sigmoid**: 从 "预测每个位置是否配对" 变为 "预测每个碱基选谁配对"——天然满足约束，无需后处理 projection
2. **Triangle Update**: 用 O(L³) 的三体交互捕获配对竞争关系——如果 i 已经和 k 配对，自动抑制 i 和 j 配对
3. **保留 Flow Matching**: 离散 flow 的逐步去噪思路是好的（v1 已证明），只是需要更好的参数化和 loss

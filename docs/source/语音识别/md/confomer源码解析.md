# confomer源码解析

## 1.位置编码

```python
class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        # 创建一个形状为 (max_len, d_model) 的零矩阵，表示位置编码的数值
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        # 生成从 0 到 max_len 的位置张量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算正弦和余弦函数的频率项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # 计算正弦和余弦值并填充到位置编码矩阵的偶数和奇数索引位置
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        # 返回指定长度的位置编码
        return self.pe[:, :length]
```



这段代码计算了位置编码中的频率项 `div_term`，其算法依据来自于 "Attention Is All You Need" 论文中提出的一种基于正弦和余弦函数的位置编码方法。具体来说，论文中建议使用以下公式来计算位置编码的每个维度的值：
$$
\text{PE}(pos, 2i) = \sin\left(\frac{{\text{pos}}}{{\text{power}(10000,\frac{{2i}}{{\text{d_model}}})}}\right)
$$

$$
\text{PE}(pos, 2i+1) = \cos\left(\frac{{\text{{pos}}}}{{\text{power}(10000, \frac{{2i+1}}{{\text{d_model}}})}}\right)
$$


其中，$\text{pos}$表示位置，$i$ 表示维度，$d\_model$ 表示模型的维度。而 $\text{power}(10000,x)$ 表示 $10000^x$。

在这段代码中，通过使用 `torch.arange(0, d_model, 2).float()` 生成一个包含 0 到 `d_model` 之间偶数索引的浮点数张量。然后，对这个张量乘以 $-\frac{\log(10000.0)}{{\text{d_model}}}$，并使用 `torch.exp` 进行指数运算，得到了 `div_term` 张量。

## 2. 注意力机制

深度学习中的注意力机制（Attention Mechanism）是一种模仿人类视觉和认知系统的方法，它允许神经网络在处理输入数据时集中注意力于相关的部分。通过引入注意力机制，神经网络能够自动地学习并选择性地关注输入中的重要信息，提高模型的性能和泛化能力。

**最典型的注意力机制包括自注意力机制、空间注意力机制和时间注意力机制**。

### 2.1 自注意力

1. **计算 Query、Key 和 Value 的线性变换：**
   - 对于给定的 $Query (Q)$, $Key (K)$, 和 $Value (V)$，通过线性变换得到 $Q' = QW_Q$, $K' = KW_K$, $V' = VW_V$。
   - 其中，$W_Q$, $W_K$, 和 $W_V$ 是学习的权重矩阵。

2. **计算、缩放注意力分数（Scaled Dot-Product Attention）：**
   - 计算注意力分数 $(A)$，其中 $A = \frac{Q' \cdot K'^T}{{\sqrt{d_k}}}$，$d_k$ 是每个头的维度（head_dim）。
   - $(Q')$ 和 $K'^T$ 分别表示 Query 和 Key 的转置矩阵。

3. **应用 Softmax 归一化**
   - 对每个注意力分数矩阵 \(A\) 的每行应用 $Softmax$ 函数，得到权重矩阵 $(W)$。
   - $W_{ij}$ 表示$ Query(i)$ 对 $Key (j)$ 的注意力权重。

4. **计算加权和（Weighted Sum）：**
   - 使用注意力权重 \(W\) 对 Value 矩阵进行加权和，得到最终的输出矩阵。
   - $O = W \cdot V'$

上述过程可以用以下数学表示：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V
$$
其中，
- $Q, K, V$ 分别是 Query、Key、Value 矩阵，
- $Q \cdot K^T$ 表示矩阵乘法，
- $\sqrt{d_k}$是缩放因子，$d_k$ 是每个头的维度。

```python
from math import sqrt

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False) # Q、K的维度一致
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k) # 为了规范Q@K的乘积的方差范围

    def forward(self, x):
        # x: (batch, n, dim_in) ——> (批量大小, 时序长度, 特征维度)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)
        return att
```

### 2.2 多头注意力



1. **线性变换：**
    - 输入 $Query Q$*,Key K*$, 和 $Value *V* 首先分别通过线性变换，得到多组 $′*Q*′$, $′*K*′$′*V*′$。
2. **分割成多头：**
    - 将每组 ′*Q*′, ′*K*′, ′*V*′ 分割成多个头（通常是 8 或 16 头）。
    - 分割的方式是将线性变换后的向量按照头的数量进行切片。
3. **并行计算注意力：**
    - 对每个头独立地计算注意力分数和注意力权重。
    - 每个头的计算独立，可以并行进行。
4. **拼接多头输出：**
    - 将每个头的注意力输出拼接在一起。
5. **线性变换得到最终输出：**
    - 将拼接后的输出再进行一次线性变换，得到最终的多头注意力输出。

```python
import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 线性变换矩阵
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # 最终输出的线性变换
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        # 获取批次大小和序列长度
        batch_size, seq_len, _ = query.shape

        # 将输入通过线性变换进行投影
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # 将每个头的维度进行分割
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # 将头移动到序列维度后进行点积注意力计算
        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        if mask is not None:
            # 满足条件（即 mask == 0 为 True 的位置）的元素替换为指定的值 float("-1e20")。这是一个很小的负数，通常用于表示负无穷大。在注意力机制中，将这些位置替换为负无穷大有助于在 Softmax 操作时将这些位置的权重变得非常接近于零。
            energy = energy.masked_fill(mask == 0, float("-1e20"))
		
        attention = F.softmax(energy / (self.d_model ** (1 / 2)), dim=3)

        # 将头移回到头维度后合并
        x = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(
            batch_size, seq_len, self.d_model
        )

        # 通过全连接层进行最终的投影
        x = self.fc_out(x)
        return x

```

```python
import torch
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 线性变换矩阵
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        # 最终输出的线性变换
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        batch_size, seq_len, _ = query.size()

        # 线性变换
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # 头的分割
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # 调整维度顺序 nqhd--->nhdq
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # 计算注意力得分
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 注意力权重
        attention = F.softmax(energy, dim=-1)

        # 使用注意力权重对值进行加权求和
        x = torch.matmul(attention, V)

        # 调整维度，合并多头
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, -1)

        # 最终的线性变换
        x = self.fc_out(x)

        return x

```

### 2.3 多头相对自注意力

```python
class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)
```

其中RelativeMultiHeadAttention实现如下：

```python
class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    具有相对位置编码的多头自注意力机制。
    这个概念是在"Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"中提出的。

    Args:
        d_model (int): 模型的维度
        num_heads (int): 注意力头的数量
        dropout_p (float): 丢弃概率

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): 包含查询向量的张量
        - **key** (batch, time, dim): 包含键向量的张量
        - **value** (batch, time, dim): 包含值向量的张量
        - **pos_embedding** (batch, time, dim): 位置嵌入张量
        - **mask** (batch, 1, time2) 或 (batch, time1, time2): 包含要屏蔽的索引的张量

    Returns:
        - **outputs**: 由相对多头自注意力模块产生的张量。
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        # 用于投影query, key, value和位置编码的线性层
        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        # 初始化相对位置编码的参数
        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        # 用于投影输出的线性层
        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)
        # 使用线性层对query, key, value和位置编码进行投影，并将结果reshape为多头
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        # 计算内容得分和相对位置得分
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)
        # 合并内容得分和相对位置得分，除以sqrt(d_model)
        score = (content_score + pos_score) / self.sqrt_dim
        # 如果有mask，应用mask
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)
        # 计算注意力权重并应用dropout
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)
        # 计算上下文，并将结果reshape回(d_model,)的形状
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)
        # 通过线性层投影得到最终输出
        return self.out_proj(context)
```

## 3. 卷积模块

### 3.1 GLU门控单元

1. **初始化：**
    - 模块使用参数`dim`进行初始化，该参数表示沿着哪个维度将输入张量分成两部分。
2. **前向方法：**
    - `forward`方法接受一个输入张量`inputs`。
    - 输入张量沿着指定的维度（`dim`）被分成两部分。使用`chunk`方法实现此目的，并返回两个张量：`outputs`和`gate`。
    - `outputs`张量表示主要的信息内容，而`gate`张量表示门控信息。
3. **门控机制：**
    - 门控机制使用Sigmoid函数（`gate.sigmoid()`）实现。Sigmoid函数将值压缩在0和1之间。
    - `gate`张量经过Sigmoid函数，产生在0和1之间的值。这些值用于调制`outputs`张量。
4. **输出计算：**
    - 最终的输出通过将`outputs`张量与Sigmoid激活的`gate`张量进行逐元素乘法得到（`outputs * gate.sigmoid()`）。
    - 此操作有效地根据`gate`张量中的值“控制”了`outputs`张量中的信息。`gate`张量中的较大值允许更多信息通过，而较小值抑制信息。

```
class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()
```

### 3.2 Swish函数

其定义如下：
$$
\text{Swish}(x) = x \cdot \text{sigmoid}(\beta x)
$$
其中，\(\text{sigmoid}\) 是Sigmoid函数，\(\beta\) 是一个可调参数。Swish函数的优点之一是在深度神经网络中通常表现良好。

```
import torch
import torch.nn.functional as F

class Swish(torch.nn.Module):
    def __init__(self, beta: float = 1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
```
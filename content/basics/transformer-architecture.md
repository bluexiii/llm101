---
title: "Transformer架构详解"
description: "深入理解Transformer架构，这是现代大型语言模型的基础"
date: 2024-01-01
layout: "basics"
weight: 2
tags: ["transformer", "architecture", "attention", "llm"]
categories: ["basics"]
level: "intermediate"
---

# Transformer架构详解

Transformer架构是2017年由Google在论文《Attention Is All You Need》中提出的，它彻底改变了自然语言处理领域，成为现代大型语言模型的基础架构。

## 为什么需要Transformer？

在Transformer之前，序列建模主要依赖循环神经网络（RNN）和长短期记忆网络（LSTM）。这些模型存在以下问题：

- **并行化困难**: RNN需要按顺序处理序列，无法充分利用现代硬件的并行计算能力
- **长距离依赖**: 随着序列长度增加，RNN难以捕获远距离的依赖关系
- **梯度消失**: 深层RNN容易出现梯度消失问题

Transformer通过引入**自注意力机制**解决了这些问题。

## Transformer的核心组件

### 1. 自注意力机制（Self-Attention）

自注意力是Transformer的核心创新。它允许模型在处理序列中的每个位置时，同时关注序列中的所有其他位置。

#### 注意力计算过程

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    """
    计算注意力权重
    Q: Query矩阵 (batch_size, seq_len, d_k)
    K: Key矩阵 (batch_size, seq_len, d_k)
    V: Value矩阵 (batch_size, seq_len, d_v)
    """
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用mask（如果提供）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 应用softmax得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 计算输出
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

#### 多头注意力（Multi-Head Attention）

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换并重塑为多头
        Q = self.w_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output, attention_weights = attention(Q, K, V, mask)
        
        # 连接多头输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 最终线性变换
        output = self.w_o(attention_output)
        
        return output, attention_weights
```

### 2. 位置编码（Positional Encoding）

由于Transformer没有循环结构，需要显式地添加位置信息。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### 3. 前馈神经网络（Feed-Forward Network）

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

### 4. 编码器层（Encoder Layer）

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 5. 解码器层（Decoder Layer）

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

## 完整的Transformer模型

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, src, src_mask):
        src = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_mask)
            
        return src
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        tgt = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt, enc_output, src_mask, tgt_mask)
            
        return tgt
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = self.final_layer(dec_output)
        
        return output
```

## Transformer的优势

1. **并行化**: 所有位置可以同时计算，充分利用GPU并行能力
2. **长距离依赖**: 自注意力机制可以直接连接任意两个位置
3. **可解释性**: 注意力权重提供了模型关注点的可视化
4. **可扩展性**: 架构简单，易于扩展到更大的模型

## 实际应用中的变体

### 仅编码器模型（BERT）
- 用于理解任务：文本分类、命名实体识别、问答
- 双向注意力，可以同时看到前后文

### 仅解码器模型（GPT）
- 用于生成任务：文本生成、对话、代码生成
- 单向注意力，只能看到前面的内容

### 编码器-解码器模型（T5、BART）
- 用于序列到序列任务：翻译、摘要、问答
- 编码器处理输入，解码器生成输出

## 学习建议

1. **理解数学原理**: 深入理解注意力机制的计算过程
2. **动手实现**: 尝试从零开始实现一个简单的Transformer
3. **可视化注意力**: 使用工具可视化注意力权重
4. **阅读原始论文**: 仔细阅读《Attention Is All You Need》论文
5. **实践应用**: 使用Hugging Face Transformers库进行实际应用

## 下一步学习

掌握Transformer架构后，建议继续学习：

- [注意力机制的深入理解](/basics/attention-mechanism/)
- [预训练与微调技术](/basics/pretraining-finetuning/)
- [提示工程实践](/basics/prompt-engineering/)

## 相关资源

- [原始论文: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Transformers教程](https://huggingface.co/course)
- [PyTorch Transformer实现](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) 
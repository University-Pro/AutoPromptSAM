import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAtten(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, head, mlp_hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = SelfAtten(dim, head)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 自注意力 + 残差连接 + LayerNorm
        x = x + self.dropout(self.attention(self.norm1(x)))
        # MLP + 残差连接 + LayerNorm
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x
    
def test_transformer_block():
    # 模拟输入：Batch=2, 序列长度=16, 特征维度=128
    B, N, C = 2, 16, 128
    x = torch.randn(B, N, C)

    # 创建 TransformerBlock 实例
    transformer = TransformerBlock(dim=128, head=4, mlp_hidden_dim=256, dropout=0.1)

    # 前向传播
    output = transformer(x)

    # 输出结果
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

# 测试
test_transformer_block()
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import MultiHeadAttention
from model.util import LayerNorm, FeedForward

"""
class:
    RegressiveTransformerBlock()

    RegressiveTransformer()
"""

class RegressiveTransformerBlock(nn.Module):
    """
        A block of Transformer on Regression task.
    """
    def __init__(self, d_model, d_hidden, num_head, dropout, eps=1e-5):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_head=num_head)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model, eps)
        
        self.ffn = FeedForward(d_model, d_hidden)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model, eps)

    def forward(self, x, mask=None):
        # input: (B, T, C)
        x = x + self.dropout1(self.attention(x, x, x, mask))
        x = self.norm1(x)

        x = x + self.dropout2(self.ffn(x))
        x = self.norm2(x)
        return x

class RegressiveTransformer(nn.Module):
    """
        Transformer for regression task by changing Embedding and Softmax to MLP.
    """
    def __init__(self, N, d_model, d_hidden, num_head, max_seq_len=100, dropout=0.1, eps=1e-5):
        super().__init__()
        self.pos_emb = nn.Linear(1, d_model)
        self.proj = nn.ModuleList([
            nn.Linear(1, d_model)
        ])

        self.blocks = nn.ModuleList([
            RegressiveTransformerBlock(d_model, d_hidden, num_head, dropout, eps) for _ in range(N)
        ])

        self.reproj = nn.ModuleList([
            nn.Linear(d_model, 1)
        ])

    def forward(self, x, mask=None):
        # input: (B, T)

        # pos = self.
        # (B, T, C)
        x = self.proj(x)

        for block in self.blocks:
            x = self.block(x, mask)
        
        # (B, T, 1)
        x = self.reproj(x)
        
        return x
        
if __name__ == "__main__":
    pass
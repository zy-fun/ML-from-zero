import torch
import torch.nn as nn
import torch.nn.functional as F
from model.util import *
from model.attention import *

"""
class:
    EncoderBlock(self, d_model, d_hidden, num_head, dropout=0.1, eps=1e-5)
    DncoderBlock(self, d_model, d_hidden, num_head, dropout=0.1, eps=1e-5)
    Encoder(self, N, d_model, d_hidden, num_head, dropout=0.1, eps=1e-5)
    Decoder(self, N, d_model, d_hidden, num_head, dropout=0.1, eps=1e-5)
"""

# the dropout layer setting is a bit confusing
class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_hidden, num_head, dropout=0.1, eps=1e-5):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_head)
        self.ffn = FeedForward(d_model, d_hidden)
        self.norm1 = LayerNorm(d_model, eps)
        self.norm2 = LayerNorm(d_model, eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = x + self.dropout1(self.attention(x, x, x, mask))
        out = self.norm1(out)

        out = out + self.dropout2(self.ffn(out))
        out = self.norm2(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_hidden, num_head, dropout=0.1, eps=1e-5):
        super().__init__()
        self.attention1 = MultiHeadAttention(d_model, num_head)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model, eps)

        self.attention2 = MultiHeadAttention(d_model, num_head) # masked
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model, eps)

        self.ffn = FeedForward(d_model, d_hidden)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(d_model, eps)

    def forward(self, x, enc, mask=None):
        x = x + self.dropout1(self.attention1(x, x, x, mask=mask))  # only attention1 enables mask
        x = self.norm1(x)

        x = x + self.dropout2(self.attention2(q=x, k=enc, v=enc))
        x = self.norm2(x)

        x = x + self.dropout3(self.ffn(x))
        x = self.norm3(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, N, d_model, d_hidden, num_head, dropout=0.1, eps=1e-5):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, d_hidden, num_head, dropout, eps) for _ in range(N)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, N, d_model, d_hidden, num_head, dropout=0.1, eps=1e-5):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, d_hidden, num_head, dropout, eps) for _ in range(N)
        ])

    def forward(self, x, enc, mask=None):
        for block in self.blocks:
            x = block(x, enc, mask=mask)
        return x

if __name__ == "__main__":
    print("main function of transformer.py")
    pass
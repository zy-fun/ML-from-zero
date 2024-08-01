import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
from attention import *

# the dropout layer setting is a bit confusing
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, d_hidden, num_head, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_head)
        self.ffn = FeedForward(d_model, d_hidden, dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = x + self.attention(x, x, x, mask)
        out = self.norm1(out)
        out = out + self.ffn(out)
        out = self.norm2(out)
        return out

if __name__ == "__main__":
    pass
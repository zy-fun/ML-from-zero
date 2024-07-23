import torch
import torch.nn as nn

# An Attention Head is indeed a complete attention layer.
class AttentionHead(nn.Module):
    def __init__(self, embd_size, head_size=16, dropout=0.1):
        super().__init__()
        self.weighted_Q = nn.Linear(embd_size, head_size)
        self.weighted_K = nn.Linear(embd_size, head_size)
        self.weighted_V = nn.Linear(embd_size, head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input size: (B, T, embd_size)
        Q = self.weighted_Q(x)  # (B, T, head_size)
        K = self.weighted_K(x)  # (B, T, head_size)
        attention_matrix = Q @ K.permute(0, 2, 1)   # (B, T, T)

        V = self.weighted_V(x)  # (B, T, head_size)
        output = attention_matrix @ V
        output = self.dropout(output)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embd_size, head_size=8, n_head=4, dropout=0.1):
        super().__init__()
        self.heads = torch.nn.ModuleList([AttentionHead(embd_size, head_size, dropout=dropout) 
                                          for _ in range(n_head)])  # use ModuleList to support nn.Module.to() method

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return output

if __name__ == "__main__":
    b = 32
    t = 100
    embd_size = 64
    device = 'cuda'
    x = torch.ones(b, t, embd_size).to(device)
    # head = AttentionHead(embd_size).to(device)
    # y = head(x) 
    attention = MultiHeadAttention(embd_size).to(device)
    y = attention(x)
    print(x.shape, y.shape)
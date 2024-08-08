import torch
import torch.nn as nn
import torch.nn.functional as F
from model.util import *
from model.attention import *

"""
class:
    EncoderBlock(self, d_model, d_ff, num_head, dropout=0.1, eps=1e-5)
    DncoderBlock(self, d_model, d_ff, num_head, dropout=0.1, eps=1e-5)

    Encoder(self, N, vocab_size, d_model, d_ff, num_head, dropout=0.1, eps=1e-5)
    Decoder(self, N, vocab_size, d_model, d_ff, num_head, dropout=0.1, eps=1e-5)

    Transformer(self, N, vocab_size, d_model, d_ff, num_head, dropout=0.1, eps=1e-5)
"""

# the dropout layer setting is a bit confusing
class EncoderBlock(nn.Module):
    """
        a block of transformer encoder
    """
    def __init__(self, d_model, d_ff, num_head, dropout=0.1, eps=1e-5):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_head)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model, eps)
        self.norm2 = LayerNorm(d_model, eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        out = x + self.dropout1(self.attention(x, x, x))
        out = self.norm1(out)

        out = out + self.dropout2(self.ffn(out))
        out = self.norm2(out)
        return out


class DecoderBlock(nn.Module):
    """
        a block of transformer decoder
    """
    def __init__(self, d_model, d_ff, num_head, dropout=0.1, eps=1e-5):
        super().__init__()
        self.attention1 = MultiHeadAttention(d_model, num_head)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model, eps)

        self.attention2 = MultiHeadAttention(d_model, num_head) # masked
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model, eps)

        self.ffn = FeedForward(d_model, d_ff)
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


# the pos_emb here is different of the original paper, I replace it with a simple nn.Embedding(learnable)
class Encoder(nn.Module):
    """
        Transformer's encoder
    """
    def __init__(self, N, vocab_size, d_model, d_ff, num_head, max_seq_len=500, dropout=0.1, eps=1e-5):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, d_ff, num_head, dropout, eps) for _ in range(N)
        ])

    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        x = self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device))

        for block in self.blocks:
            x = block(x)
        return x


# CrossEntropy of torch includes softmax operation
# so Decoder implemented here won't need softmax layer
class Decoder(nn.Module):
    """
        Transformer's decoder
    """
    def __init__(self, N, vocab_size, d_model, d_ff, num_head, max_seq_len=500, dropout=0.1, eps=1e-5):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, d_ff, num_head, dropout, eps) for _ in range(N)
        ])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc, mask=None):
        # x: (B, T)
        B, T = x.shape
        x = self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device))

        # enc: (B, T, C)
        for block in self.blocks:
            x = block(x, enc, mask=mask)

        # logits: (B, T, V)
        logits = self.linear(x) 

        return logits
    

class Transformer(nn.Module):
    """
        an implemention of simplified Transformer
    """
    def __init__(self, N, vocab_size, d_model, d_ff, num_head, max_seq_len=500, dropout=0.1, eps=1e-5):
        super().__init__()
        self.encoder = Encoder(N=N, 
                               vocab_size=vocab_size, 
                               d_model=d_model, 
                               d_ff=d_ff,
                               num_head=num_head,
                               max_seq_len=max_seq_len,
                               dropout=dropout,
                               eps=eps)
        self.decoder = Decoder(N=N, 
                               vocab_size=vocab_size, 
                               d_model=d_model, 
                               d_ff=d_ff,
                               num_head=num_head,
                               max_seq_len=max_seq_len,
                               dropout=dropout,
                               eps=eps)

    def forward(self, x, enc, mask=None):
        enc = self.encoder(enc)
        logits = self.decoder(x, enc, mask=mask)
        return logits

if __name__ == "__main__":
    print("main function of transformer.py")
    N = 6
    batch_size = 4
    seq_len = 100
    vocab_size = 1000
    d_model = 512
    d_ff = 2048
    num_head = 8
    dropout = 0.1
    device = 'cuda'

    enc = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    data = torch.randint(0, vocab_size, (batch_size, seq_len + 1)).to(device)
    x, y = data[:, :seq_len], data[:, 1:]
    model = Transformer(N, vocab_size, d_model, d_ff, num_head, dropout=dropout).to(device)

    logits = model(x, enc)
    print(logits.shape)
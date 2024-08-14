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
    def __init__(self, N, d_feature, d_model, d_hidden, num_head, dropout=0.1, eps=1e-5):
        super().__init__()
        # It seens that the pos_emb could be one of the bottleneck, 
        # I add one more Linear and the loss gets smaller.
        self.pos_emb = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Linear(d_model, d_model),
        )
        
        self.proj = nn.Sequential(
            nn.Linear(d_feature, d_model),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            RegressiveTransformerBlock(d_model, d_hidden, num_head, dropout, eps) for _ in range(N)
        ])

        self.reproj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_feature),
        )

    def forward(self, x, mask=None):
        # input: (B, T, d_feature)
        B, T, _ = x.shape

        pos = torch.arange(0, T, step=1, dtype=x.dtype, device=x.device)    # (T,)
        pos = self.pos_emb(pos.unsqueeze(-1))   # (T, C)

        # (B, T, C)
        x = self.proj(x) + pos.unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)
        
        # (B, T, d_feature)
        x = self.reproj(x)
        
        return x
        
if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. define model
    N = 6
    d_feature = 4
    d_model = 32
    d_ffn = 512
    num_head = 8
    device = 'cuda'
    
    model = RegressiveTransformer(N, d_feature, d_model, d_ffn, num_head).to(device)

    # 2. define data
    batch_size = 4
    block_size = 100
    data = torch.randn((batch_size, block_size + 1, d_feature), dtype=torch.float32).to(device)
    x = data[:, :block_size, :]
    y = data[:, 1:, :]

    # 3. forward and backward
    max_iter = 2000
    lr = 8e-4
    mask = torch.triu(torch.ones((1, block_size, block_size), 
                                  dtype=torch.bool), diagonal=1).to(device)
    model.train()
    import torch.optim as optim; optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(max_iter): 
        optimizer.zero_grad()           # clear the grad
        output = model(x, mask)         # forward
        loss = F.mse_loss(output, y)    # compute loss
        loss.backward()                 # backward and compute the grad
        optimizer.step()                # update model
        print(f"loss: {loss}")
    
    print(output[0, :5])
    print(y[0, :5])
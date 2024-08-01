import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class:
    MultiHeadAttention(d_model, num_head, dropout)
"""

# To do:
# mask waits to implement
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        self.num_head = num_head
        assert d_model % num_head == 0
        self.head_size = d_model // num_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim= -1)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = q.shape
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # (batch_size, num_head, seq_len, head_size)
        q = q.reshape(-1, seq_len, self.num_head, self.head_size).transpose(1, 2)
        k = k.reshape(-1, seq_len, self.num_head, self.head_size).transpose(1, 2)
        v = v.reshape(-1, seq_len, self.num_head, self.head_size).transpose(1, 2)

        # (batch_size, num_head, seq_len, seq_len)
        score = q @ k.transpose(2, 3)
        score = score / torch.sqrt(torch.tensor(self.head_size))  # scaled dot

        # apply mask
        if mask is not None:
            pass
        
        # normalize attention score matrix for each row (dim= -1)
        score = self.softmax(score)

        output = score @ v  # (batch_size, num_head, seq_len, head_size)
        output = output.transpose(1, 2).reshape(-1, seq_len, self.num_head * self.head_size)    # (batch_size, seq_len, d_model)
        
        return output

if __name__ == "__main__":
    batch_size = 32
    seq_len = 100
    d_model = 64
    device = 'cuda'
    
    x = torch.ones(batch_size, seq_len, d_model).to(device)
    attention = MultiHeadAttention(d_model, num_head=8).to(device)
    y = attention(x,x,x)
    # print(x.shape, y.shape)
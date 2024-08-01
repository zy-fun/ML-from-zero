import torch
from model.attention import *

if __name__ == "__main__":
    b = 32
    t = 100
    embd_size = 64
    device = 'cuda'
    x = torch.ones(b, t, embd_size).to(device)
    attention = MultiHeadAttention(embd_size, 8).to(device)
    y = attention(x, x, x) 
    print(x.shape, y.shape)
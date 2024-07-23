import torch
from model.model import *

if __name__ == "__main__":
    b = 32
    t = 100
    embd_size = 64
    device = 'cuda'
    x = torch.ones(b, t, embd_size).to(device)
    head = AttentionHead(embd_size).to(device)
    y = head(x) 
    print(x.shape, y.shape)
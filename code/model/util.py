import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, eps):
        pass

    def forward(self, x):
        mean = torch.mean(x)
        
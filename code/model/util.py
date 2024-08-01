import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class:
    LayerNorm(self, n_embd, eps)
'''

# This LayerNorm is rewrited for learning

# Note: 'unbiased' parameter of tensor.var(): 
# True for denominator equals N-1
# and False for denominator equals N

class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, x: torch.tensor):
        # x: [B, T, C]
        # x = x.float()   # not sure
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)

        out = out * self.gamma + self.beta
        return out
        
if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3],
                      [4, 8, 12]], dtype=torch.float32)
    rewrited_layernorm = LayerNorm(3, 1e-5)
    rewrited_out = rewrited_layernorm(x)
    torch_layernorm = nn.LayerNorm(x.shape[-1]) 
    torch_out = torch_layernorm(x)
    print(rewrited_out)
    print(torch_out)
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class:
    FeedForward(self, d_model, d_hidden)
    LayerNorm(self, d_model, eps)

function:
    get_padding_mask(data, padding)
    get_decoder_mask(data)
'''

# two linear layers with one relu
# linear(x) equals x @ linear.weight.T + linear.bias
# I decide to discard the dropout operation in ffn
class FeedForward(nn.Module):
    """
        FFN of transformer, maybe it shouldn't be in util.py
    """
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)  # dropout will divide the tensor by (1-p) to balance

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = self.dropout(x)
        return x
    

# This LayerNorm is rewrited for learning

# Note: 'unbiased' parameter of tensor.var(): 
# True for denominator equals N-1
# and False for denominator equals N
class LayerNorm(nn.Module):
    """
        an implemention of layernorm 
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.tensor):
        # x: [B, T, C]
        # x = x.float()   # not sure
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)

        out = out * self.gamma + self.beta
        return out
    
def get_padding_mask(data, padding=-1):
    """
        generate a mask to mask padding elements
        input: (B, T)
        output: (B, T, T)
        e.g:
            input:  [[1, 2, 3, -1, -1]]
            output: [[F, F, F, T, T], 
                    [F, F, F, T, T],
                    [F, F, F, T, T],
                    [T, T, T, T, T],
                    [T, T, T, T, T]]
    """
    B, T = data.shape
    
    padding_mask = torch.zeros((B, T, T), dtype=torch.bool, device=data.device)
    

    return padding_mask
        
if __name__ == "__main__":
    layernormtest = False
    ffntest = False
    mask_test = True

    # layernorm testing
    if layernormtest:
        device = 'cuda'
        x = torch.tensor([[1, 2, 3],
                        [4, 8, 12]], dtype=torch.float32).to(device)
        rewrited_layernorm = LayerNorm(3, 1e-5).to(device)
        rewrited_out = rewrited_layernorm(x)
        torch_layernorm = nn.LayerNorm(x.shape[-1]).to(device)
        torch_out = torch_layernorm(x)
        print(rewrited_out)
        print(torch_out)

    # feedforward testing
    if ffntest:
        device = 'cuda'
        feedforward = FeedForward(x.shape[-1], 2).to(device)
        out = feedforward(x)
        print(out)
        print(out.shape)

    if mask_test:
        device = 'cuda'
        x = torch.tensor([[1, 2, 3, -1, -1],
                          [1, -1, -1, -1, -1]]).to(device)
        padding_mask = get_padding_mask(x, -1)
        print(x)
        print(padding_mask)

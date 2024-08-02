import torch
from model.transformer import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=1000)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--device', type=str, default='cuda')
cfg = vars(parser.parse_args())

N = cfg['N']
batch_size = cfg['batch_size']
vocab_size = cfg['vocab_size']
seq_len = cfg['seq_len']
d_model = cfg['d_model']
d_ff = cfg['d_ff']
num_head = cfg['num_head']
dropout = cfg['dropout']
eps = cfg['eps']
device = cfg['device']

if __name__ == "__main__":
    enc = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    data = torch.randint(0, vocab_size, (batch_size, seq_len + 1)).to(device)
    x, y = data[:, :seq_len], data[:, 1:]

    model = Transformer(N, vocab_size, d_model, d_ff, num_head, dropout=dropout).to(device)

    logits = model(x, enc)
    print(logits.shape)
import torch
from model.transformer import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--device', type=str, default='cuda')
cfg = vars(parser.parse_args())

N = cfg['N']
batch_size = cfg['batch_size']
seq_len = cfg['seq_len']
d_model = cfg['d_model']
d_ff = cfg['d_ff']
num_head = cfg['num_head']
dropout = cfg['dropout']
device = cfg['device']

if __name__ == "__main__":
    x = torch.ones(batch_size, seq_len, d_model).to(device)
    encoder = Encoder(N, d_model, d_ff, num_head, dropout)
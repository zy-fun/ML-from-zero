import torch
from torch.utils.data import DataLoader
# from model.transformer import *
# from model.tokenizer import *
from model.RegressiveTransformer import RegressiveTransformer
from model.util import get_sequence_mask
from dataloader import StockDataset
import argparse
import akshare as ak
import mplfinance as mpf
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=1000)
parser.add_argument('--max_iter', type=int, default=1e6)
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

num_epoch = cfg['num_epoch']
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
    model = RegressiveTransformer(N, 4, d_model, d_ff, num_head).to(device)

    dataset = StockDataset("./dataset", start=0, end=0.01, blocksize=seq_len)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    mask = get_sequence_mask(seq_len).to(device)
    for data_x, data_y in tqdm(dataloader):
        data_x = data_x.permute(0,2,1).to(device)
        data_y = data_y.permute(0,2,1).to(device)
        break
    print(data_x.shape)
    output = model(data_x, mask)
    print(output.shape)

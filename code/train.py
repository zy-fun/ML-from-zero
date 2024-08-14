import torch
from torch.utils.data import DataLoader
from model.RegressiveTransformer import RegressiveTransformer
from model.util import get_sequence_mask
from dataloader import StockDataset
import argparse
import akshare as ak
import mplfinance as mpf
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()

# environment setting
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda')

# train setting
parser.add_argument('--max_iter', type=int, default=1e6)
parser.add_argument('--eval_iter', type=int, default=1e3)
parser.add_argument('--batch_per_iter', type=int, default=1000)
parser.add_argument('--num_epoch', type=int, default=1000)
parser.add_argument('--load_model', type=str, default=None)

# model setting
parser.add_argument('--N', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=1000)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=1e-5)

# file path setting
parser.add_argument('--dataset_dirname', type=str, default='./dataset')
parser.add_argument('--log_dirname', type=str, default='./log')
parser.add_argument('--save_dirname', type=str, default='./saved_model')
cfg = vars(parser.parse_args())

max_iter = cfg['max_iter']
eval_iter = cfg['eval_iter']
batch_per_iter = cfg['batch_per_iter']
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
    d_feature = 4   # open, close, high, low
    model = RegressiveTransformer(N, 4, d_model, d_ff, num_head).to(device)

    train_portion = 0.9
    val_portion = 0.1
    dataset = StockDataset("./dataset", start=0, end=train_portion, blocksize=seq_len)
    train = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model.train()
    for iter in range(max_iter):
        if iter % eval_iter == 0:
            pass
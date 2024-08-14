import torch
import torch.optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from model.RegressiveTransformer import RegressiveTransformer
from model.util import get_sequence_mask
from dataloader import StockDataset
import argparse
import mplfinance as mpf
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()

# environment setting
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda')

# model setting
parser.add_argument('--N', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--d_feature', type=int, default=4)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=1e-5)

# file path setting
parser.add_argument('--dataset_dirname', type=str, default='./dataset')
parser.add_argument('--log_dirname', type=str, default='./log')
parser.add_argument('--load_dirname', type=str, default='./saved_model')
cfg = vars(parser.parse_args())

seed = cfg['seed']
N = cfg['N']
seq_len = cfg['seq_len']
d_feature = cfg['d_feature']   # open, close, high, low
d_model = cfg['d_model']
d_ff = cfg['d_ff']
num_head = cfg['num_head']
dropout = cfg['dropout']
eps = cfg['eps']
device = cfg['device']
load_dirname = cfg['load_dirname']


model_name = f"N{N}-T{seq_len}-d_feature{d_feature}-d_model{d_model}-d_ff{d_ff}-num_head{num_head}-dropout{dropout}-eps{eps}"
load_path = os.path.join(load_dirname, f"{model_name}.pth")

if __name__ == "__main__":
    torch.manual_seed(seed)
    model = RegressiveTransformer(N, 4, d_model, d_ff, num_head, dropout, eps).to(device)
    model.load_state_dict(torch.load(load_path))

    train_portion = 0.01
    val_portion = 0.01

    train_set = StockDataset("./dataset", start=0, end=train_portion, blocksize=seq_len)
    train = DataLoader(train_set, batch_size=4, shuffle=True)

    val_set = StockDataset("./dataset", start=train_portion, end=train_portion+val_portion, blocksize=seq_len)
    val = DataLoader(val_set, batch_size=4, shuffle=True)

    data, target = next(train.__iter__())
    # data, target = next(val.__iter__())
    data = data.permute(0, 2, 1)

    gen_len = 100
    model.eval()
    for i in tqdm(range(100), desc='autoregressive'):
        gen = model(data[:, -seq_len: , :])
        data = torch.concat([data, gen[:, -1:, :]], dim = 1)

    stock = data[0].cpu().detach().numpy()
    stock = np.round(stock, 2)
    columns = ['Open', 'Close', 'High', 'Low']
    stock = pd.DataFrame(stock, columns=columns)
    stock.index = pd.to_datetime(stock.index)
    print(stock)
    # the generated sucks now.
    mpf.plot(stock, mav=(3,6,9), figscale=2, type='candle', style='charles', savefig='fig/gen.png')
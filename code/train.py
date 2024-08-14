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
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()

# environment setting
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda')

# train setting
parser.add_argument('--max_iter', type=int, default=int(1e4))
parser.add_argument('--eval_iter', type=int, default=int(1e2))
parser.add_argument('--save_iter', type=int, default=int(1e4))
parser.add_argument('--batch_per_iter', type=int, default=1000)
parser.add_argument('--num_epoch', type=int, default=1000)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--lr', type=float, default=1e-5)

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
load_model = cfg['load_model']
lr = cfg['lr']

seq_len = cfg['seq_len']
d_model = cfg['d_model']
d_ff = cfg['d_ff']
num_head = cfg['num_head']
dropout = cfg['dropout']
eps = cfg['eps']
device = cfg['device']
save_dirname = cfg['save_dirname']

d_feature = 4   # open, close, high, low

model_name = f"N{N}-T{seq_len}-d_feature{d_feature}-d_model{d_model}-d_ff{d_ff}-num_head{num_head}-dropout{dropout}-eps{eps}"
save_path = os.path.join(save_dirname, f"{model_name}.pth")

if __name__ == "__main__":
    # 1. define the model (optional: load saved model)
    model = RegressiveTransformer(N, 4, d_model, d_ff, num_head, dropout, eps).to(device)
    if load_model:
        model.load_state_dict(torch.load(save_path))
    print("-" * 30 + f"date time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}" + "-" * 30)
    print()
    print(f"model name:\t{model_name}")

    # 2. load dataset
    train_portion = 0.01
    val_portion = 0.01

    train_set = StockDataset("./dataset", start=0, end=train_portion, blocksize=seq_len)
    train = DataLoader(train_set, batch_size=4, shuffle=True)

    val_set = StockDataset("./dataset", start=train_portion, end=train_portion+val_portion, blocksize=seq_len)
    val = DataLoader(val_set, batch_size=4, shuffle=True)
    
    # 3. train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    mean_loss = 0
    for itera in tqdm(range(max_iter)):
        optimizer.zero_grad() 
        data, target = next(train.__iter__())
        data = data.permute(0, 2, 1)
        target = target.permute(0, 2, 1)
        predict = model(data)
        loss = mse_loss(predict, target)
        loss.backward()
        optimizer.step()

        mean_loss += loss

        if itera % eval_iter == 0:
            torch.save(model.state_dict(), save_path)
            print(f"iter {itera}:\ttrain loss {mean_loss / (itera + 1)}")
            pass
import akshare as ak
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
import torch
from torch.utils.data import Dataset, DataLoader

"""
class:
    StockDataset(dirname, start, end, blocksize)

function:
    download_stock(dirname)
"""

class StockDataset(Dataset):
    """
        A dataset of stock datas

        Structure of data:
            dataset/
                stock_list.csv
                bj430017.csv
                ...
                sz301608.csv
    """
    def __init__(self, dirname, start = 0, end = 1, blocksize=100):
        super().__init__()
        self.dirname = dirname
        self.stock_list = pd.read_csv(os.path.join(dirname, 'stock_list.csv'))

        # filter data out of range(start, end)
        self.stock_list = self.stock_list.iloc[int(start * len(self.stock_list)) : 
                                               int(end * len(self.stock_list))]
        self.symbols = self.stock_list['代码'].tolist()
        self.blocksize = blocksize
        
        self.data = []
        for i, symbol in enumerate(tqdm(self.symbols, desc="loading dataset")):
            # load dataframe from csv
            stock_df = pd.read_csv(os.path.join(dirname, f"{symbol}.csv"))

            # fetch price column and convert dataframe to numpy array
            stock_np = stock_df[['开盘', '收盘', '最高', '最低']].to_numpy().T
            self.data.append(stock_np)

            # insert seperator
            self.data.append(np.array([[0.0] for _ in range(4)]))
        self.data = torch.tensor(np.concatenate(self.data, axis=-1), device='cuda')

    def __len__(self):
        return self.data.shape[1] - self.blocksize

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.data[:, idx: idx + self.blocksize], self.data[:, idx + 1: idx + self.blocksize + 1]

def download_stock(dirname='./dataset'):
    """
        Download stock data through akshare to given directory.
    """
    # 1. get the symbol list of stocks
    stock_list_path = os.path.join(dirname, 'stock_list.csv')

    if os.path.exists(stock_list_path):
        stock_list = pd.read_csv(stock_list_path)
    else:
        stock_list = ak.stock_zh_a_spot()   # may fail due to connection problem
        stock_list.to_csv(stock_list_path, index=False) 

    # 2. download stock data for every symbol
    for symbol in tqdm(stock_list['代码']):
        path = os.path.join(dirname, 
                            f"{symbol}.csv")
        if os.path.exists(path):
            continue
        else:
            # adjust 'qhq' means for '前复权'
            stock_data = ak.stock_zh_a_hist(symbol=symbol[2:], period="daily", adjust="qfq")
            stock_data.to_csv(path, index=False)

if __name__ == "__main__":
    dirname = './dataset'

    # download_stock(dirname)

    start = 0
    end = 0.01
    dataset = StockDataset(dirname, start, end)
    train = DataLoader(dataset, batch_size=1000, shuffle=False)
    for i, (data_x, data_y) in enumerate(train):
        continue
    print(data_x)
    print(data_y)
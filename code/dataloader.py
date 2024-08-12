import akshare as ak
import os
import pandas as pd
from tqdm import tqdm
import pdb
from torch.utils.data import Dataset

"""
class:
    StockDataset(dirname, lazy_load)

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
    def __init__(self, dirname, lazy_load=True):
        super().__init__()
        self.dirname = dirname
        self.stock_list = pd.read_csv(os.path.join(dirname, 'stock_list.csv'))
        self.symbols = self.stock_list['代码'].tolist()
        
        self.data = [None for _ in range(len(self.symbols))]
        if not lazy_load:
            for i, symbol in enumerate(self.symbols):
                stock = pd.read_csv(os.path.join(dirname, f"{symbol}.csv"))
                self.data[i] = stock

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, idx):
        if self.data[idx] is None:
            symbol = self.symbols[idx]
            stock = pd.read_csv(os.path.join(dirname, f"{symbol}.csv"))
            self.data[idx] = stock
        return self.data[idx]

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

    ratios = [.9, .05, .05]
    dataset = StockDataset(dirname)
    import torch
    from torch.utils.data import random_split
    torch.manual_seed(42)
    train, val, test = random_split(dataset, ratios)

    print(len(dataset))
    print(len(train), len(val), len(test))
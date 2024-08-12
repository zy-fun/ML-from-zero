import akshare as ak
import os
import pandas as pd
from tqdm import tqdm
import pdb

if __name__ == "__main__":
    # 1. get the symbol list of stocks
    stock_list_path = './dataset/stock_list.csv'

    if os.path.exists(stock_list_path):
        stock_list = pd.read_csv(stock_list_path)
    else:
        stock_list = ak.stock_zh_a_spot()   # may fail due to connection problem
        stock_list.to_csv(stock_list_path, index=False) 

    # 2. download stock data for every symbol
    for symbol in tqdm(stock_list['代码']):
        if os.path.exists(f"./dataset/{symbol}.csv"):
            continue
        else:
            # adjust 'qhq' means for '前复权'
            stock_data = ak.stock_zh_a_hist(symbol=symbol[2:], period="daily", adjust="qfq")
            stock_data.to_csv(f"./dataset/{symbol}.csv", index=False)
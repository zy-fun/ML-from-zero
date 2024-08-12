import akshare as ak
import os
import pandas as pd
import pdb

if __name__ == "__main__":
    # 1. get the code list of stocks
    stock_list_path = './dataset/stock_list.csv'

    if os.path.exists(stock_list_path):
        stock_list = pd.read_csv(stock_list_path)
    else:
        stock_list = ak.stock_zh_a_spot()   # may fail due to connection problem
        stock_list.to_csv(stock_list_path, index=False) 

    code_list = stock_list['代码']

    # 2. wait to do

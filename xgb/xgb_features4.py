
import numpy as np
import pandas as pd

from itertools import product
from sklearn.preprocessing import LabelEncoder

import time
import sys
import gc
import pickle

matrix = pd.read_pickle('data_new_features3.pkl')
ts = time.time()
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
time.time() - ts


ts = time.time()
matrix = matrix[matrix.date_block_num > 11]
time.time() - ts


ts = time.time()
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)
time.time() - ts


# NB clip target here
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .clip(0,20) 
                                .astype(np.float16))


matrix.to_pickle('data_new.pkl')

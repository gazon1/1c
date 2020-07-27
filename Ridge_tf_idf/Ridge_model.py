from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import scipy
from scipy import sparse
import gc
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
SEED = 0


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage. 
        
        https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


data = pd.read_pickle("./data_linear_models_merged.pkl")
data = reduce_mem_usage(data)
data['delta_revenue_lag_1'] = data['delta_revenue_lag_1'].astype(np.float16)
data.drop(['item_category_name', 'item_name', 'item_category_id_y', 'item_category_id_x'], axis=1, inplace=True)

time_st = time.time()
X_train_ = sparse.csr_matrix(data.loc[data.date_block_num<33].drop(['item_cnt_month'], axis=1).to_numpy(copy=False))

# sparse.save_npz("X_train_.npz", X_train_)
data_item_name_cat_train = sparse.load_npz("data_item_name_cat_train.npz")
X_train = scipy.sparse.hstack([X_train_, data_item_name_cat_train])

sparse.save_npz("X_train.npz", X_train)

## VAL
print(time.time() - time_st)
print("VAL")
time_st = time.time()

del X_train
del X_train_
del data_item_name_cat_train
gc.collect()

X_val_ = sparse.csr_matrix(data.loc[data.date_block_num==33].drop(['item_cnt_month'], axis=1).to_numpy(copy=False))
data_item_name_cat_val = sparse.load_npz("data_item_name_cat_val.npz")
X_val = scipy.sparse.hstack([X_val_, data_item_name_cat_val])
sparse.save_npz("X_val.npz", X_val)

# TEST
print(time.time() - time_st)
print("TEST")
time_st = time.time()

del X_val_
del X_val
del data_item_name_cat_val
gc.collect()

X_test_ = sparse.csr_matrix(data.loc[data.date_block_num==34].drop(['item_cnt_month'], axis=1).to_numpy(copy=False))
data_item_name_cat_test = sparse.load_npz("data_item_name_cat_test.npz")
X_test = scipy.sparse.hstack([X_test_, data_item_name_cat_test])
sparse.save_npz("X_test.npz", X_test)

print(time.time() - time_st)
print("END")

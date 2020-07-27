from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
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

time_st = time.time()
data = pd.read_pickle("./data_linear_models_merged.pkl")
data = reduce_mem_usage(data)
data['delta_revenue_lag_1'] = data['delta_revenue_lag_1'].astype(np.float16)
data.drop(['item_category_name', 'item_name', 'item_category_id_y', 'item_category_id_x'], axis=1, inplace=True)

scaler = StandardScaler(with_mean=False)
X_train = sparse.load_npz("X_train.npz")
X_train = scaler.fit_transform(X_train)
Y_train = data.loc[data.date_block_num<33,'item_cnt_month'].values

X_val = sparse.load_npz("X_val.npz")
X_val = scaler.transform(X_val)
Y_val = data.loc[data.date_block_num==33,'item_cnt_month'].values

X = sparse.vstack([X_train,X_val])
Y = np.vstack([Y_train.reshape((-1,1)), Y_val.reshape((-1,1))])

train_cv = list(range(0, X_train.shape[0]))
val_cv = list(range(X_train.shape[0], X_train.shape[0] + X_val.shape[0]))

cv = zip([train_cv], [val_cv])
parameters = {'random_state':[SEED],\
              'alpha': [1e-15, 1e-10, 1e-08, 1e-05, 1e-04, 0.001, 0.01, 1, 5, 10, 100, 1000, 1e4]}
print(time.time() - time_st)
time_st = time.time()
print("start grid search!")
gs = GridSearchCV(Ridge(), parameters, cv=cv, verbose=2)
gs.fit(X, Y)
print(gs.best_params_)
print(gs.best_score_)
print(time.time() - time_st)
print("END")

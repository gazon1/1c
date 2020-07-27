import numpy as np
import pandas as pd
import sklearn
import scipy.sparse

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

from itertools import product
from sklearn.preprocessing import LabelEncoder

import time
import sys
import gc
import pickle

from NearestNeighborsFeats import NearestNeighborsFeats

import numpy as np

skf_seed = 123
n_splits = 4

# a list of K in KNN, starts with one
k_list = [3, 8, 32]

# LOAD DATA

data = pd.read_pickle('data.pkl')

data = data[[
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'city_code',
    'item_category_id',
    'type_code',
    'subtype_code',
    'item_cnt_month_lag_1',
    'item_cnt_month_lag_2',
    'item_cnt_month_lag_3',
    'item_cnt_month_lag_6',
    'item_cnt_month_lag_12',
    'date_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2',
    'date_item_avg_item_cnt_lag_3',
    'date_item_avg_item_cnt_lag_6',
    'date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1',
    'date_shop_avg_item_cnt_lag_2',
    'date_shop_avg_item_cnt_lag_3',
    'date_shop_avg_item_cnt_lag_6',
    'date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1',
    'date_shop_cat_avg_item_cnt_lag_1',
    #'date_shop_type_avg_item_cnt_lag_1',
    #'date_shop_subtype_avg_item_cnt_lag_1',
    'date_city_avg_item_cnt_lag_1',
    'date_item_city_avg_item_cnt_lag_1',
    #'date_type_avg_item_cnt_lag_1',
    #'date_subtype_avg_item_cnt_lag_1',
    'delta_price_lag',
    'month',
    'days',
    'item_shop_last_sale',
    'item_last_sale',
    'item_shop_first_sale',
    'item_first_sale',
]]
from scipy import sparse
X = data[data.date_block_num <= 33].drop(['item_cnt_month'], axis=1).values
X = sparse.csr_matrix(X)
Y = data[data.date_block_num <= 33]['item_cnt_month'].values
# Y = sparse.csr_matrix(Y)
Y = sparse.csr_matrix(Y.reshape((-1, 1)))
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1).values
X_test = sparse.csr_matrix(X_test)

del data
gc.collect();

# Get features for test
import time
ts = time.time()

for metric in ['minkowski', 'cosine']:
    print(metric)

    # Create instance of our KNN feature extractor
    NNF = NearestNeighborsFeats(n_jobs=4, k_list=k_list, metric=metric)

    # Fit on train set
    print("fitting model...")
    NNF.fit(X, Y)
    print("model is fitted!")

    # Get features for test
    print("start predicting features for the test!")
    test_knn_feats = NNF.predict(X_test)

    # Dump the features to disk
    np.save('data/knn_feats_%s_test.npy' % metric, test_knn_feats)
print(f"finished getting features for train {ts} time consumed")



from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

# We will use two metrics for KNN
ts = time.time()
for metric in ['minkowski', 'cosine']:
    print(metric)

    # Set up splitting scheme, use StratifiedKFold
    # use skf_seed and n_splits defined above with shuffle=True
    skf = StratifiedKFold(random_state=skf_seed, n_splits=n_splits, shuffle=True)

    # Create instance of our KNN feature extractor
    # n_jobs can be larger than the number of cores
    NNF = NearestNeighborsFeats(n_jobs=4, k_list=k_list, metric=metric)

    # Get KNN features using OOF use cross_val_predict with right parameters
    preds = cross_val_predict(NNF, X, Y, cv=skf)

    # Save the features
    np.save('data/knn_feats_%s_train.npy' % metric, preds)
print(f"finished getting features for test {ts} time consumed")

import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import time
import sys
import gc
import pickle

best_params = {'bagging_frequency': 0.7, 'boosting': 'gbdt', 'colsample_bytree': 0.7, 'feature_fraction': 0.9, 'learning_rate': 0.01, 'max_depth': 10, 'metric': 'rmse', 'min_child_samples': 40, 'num_leaves': 125, 'objective': 'regression', 'reg_alpha': 0.6, 'reg_lambda': 0.2, 'subsample': 1}

data = pd.read_pickle('data_new.pkl')
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

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del data
gc.collect();
import lightgbm
lgtrain = lightgbm.Dataset(X_train, label=Y_train)
lgval = lightgbm.Dataset(X_valid, label=Y_valid)
model_lgb = lightgbm.train(best_params, lgtrain, 1000, 
                      valid_sets=[lgtrain, lgval], early_stopping_rounds=500, 
                      verbose_eval=300)

# lgb_pred = model_lgb.predict(X_test).clip(0, 20)

Y_pred = model_lgb.predict(X_valid).clip(0, 20)
Y_test = model_lgb.predict(X_test).clip(0, 20)

test  = pd.read_csv('data/test.csv').set_index('ID')
submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('xgb_submission_clipped_last.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('xgb_train_clipped_last.pickle', 'wb'))
pickle.dump(Y_test, open('xgb_test_clipped_last.pickle', 'wb'))

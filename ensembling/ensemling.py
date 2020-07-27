import numpy as np
import pandas as pd 
import sklearn
import scipy.sparse 
import lightgbm 
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import product
import time
time_st = time.time()

def make_submit(preds, file_name):
    submission = pd.DataFrame(preds,columns=['item_cnt_month'])
    submission.to_csv(f'{file_name}.csv',index_label='ID')
    print(f"Made submission to file '{file_name}'")


def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    
    return df


# sales = pd.read_csv('../readonly/final_project_data/sales_train.csv.gz')
# shops = pd.read_csv('../readonly/final_project_data/shops.csv')
# items = pd.read_csv('../readonly/final_project_data/items.csv')
# item_cats = pd.read_csv('../readonly/final_project_data/item_categories.csv')

data = pd.read_pickle("data.pkl")
# Save `date_block_num`, as we can't use them as features, but will need them to split the dataset into parts 
dates = data['date_block_num']
last_block = 34
dates_train = dates[dates <  last_block]
dates_test  = dates[dates == last_block]

to_drop_cols = ['date_block_num']
X_train = data.loc[dates <  last_block].drop(to_drop_cols, axis=1).fillna(0)
X_test =  data.loc[dates == last_block].drop(to_drop_cols, axis=1).fillna(0)

y_train = data.loc[dates <  last_block, 'item_cnt_month'].values
y_test =  data.loc[dates == last_block, 'item_cnt_month'].values

# First level models
# Test meta-features
lr = LinearRegression()
lr.fit(X_train.values, y_train)
pred_lr = lr.predict(X_test.values)

print('Test R-squared for linreg is %f' % r2_score(y_test, pred_lr))

lgb_params = {
               'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':1, 
               'min_data_in_leaf': 2**7, 
               'bagging_fraction': 0.75, 
               'learning_rate': 0.03, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0 
              }

model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), 100)
pred_lgb = model.predict(X_test)

print('Test R-squared for LightGBM is %f' % r2_score(y_test, pred_lgb))

X_test_level2 = np.c_[pred_lr, pred_lgb] 

# Train meta-features
# That is how we get target for the 2nd level dataset
level2_dates = [27, 28, 29, 30, 31, 32, 33]
y_train_level2 = y_train[dates_train.isin(level2_dates)]

# Now fill `X_train_level2` with metafeatures
stack1, stack2 = [], []
for cur_block_num in tqdm(level2_dates, desc='Making meta-features for train set'):
    '''
        1. Split `X_train` into parts
           Remember, that corresponding dates are stored in `dates_train` 
        2. Fit linear regression 
        3. Fit LightGBM and put predictions          
        4. Store predictions from 2. and 3. in the right place of `X_train_level2`. 
           You can use `dates_train_level2` for it
           Make sure the order of the meta-features is the same as in `X_test_level2`
    '''      
    _train_X = X_train[dates_train.isin(list(range(0, cur_block_num)))]
    _train_Y = y_train[dates_train.isin(list(range(0, cur_block_num)))]
    _test_X = X_train[dates_train.isin([cur_block_num])].values
    
    lr = LinearRegression()
    lr.fit(_train_X, _train_Y)
    pred_lr = lr.predict(_test_X)
    
    model = lgb.train(lgb_params, lgb.Dataset(_train_X, label=_train_Y), 100)
    pred_lgb = model.predict(_test_X)
    
    stack1.append(pred_lr)
    stack2.append(pred_lgb)
    
X_train_level2 = np.c_[np.hstack(stack1), np.hstack(stack2)]

# Ensembling with simple convex mix
alphas_to_try = np.linspace(0, 1, 1001)
def mix(alpha, X=X_train_level2):
    return X[:, 0] * alpha + X[:, 1] * (1. - alpha)

r2_train_simple_mix = -np.inf
best_alpha = 0
for alpha in alphas_to_try:
    r2 = r2_score(y_train_level2, mix(alpha))
    if r2_train_simple_mix < r2:
        r2_train_simple_mix = r2
        best_alpha = alpha

print('Best alpha: %f; Corresponding r2 score on train: %f' % (best_alpha, r2_train_simple_mix))

test_preds = mix(best_alpha, X_test_level2)
r2_test_simple_mix = r2_score(y_test, test_preds)


print('Test R-squared for simple mix is %f' % r2_test_simple_mix)
make_submit(test_preds, "linear_mix_ensemble")

# Stacking
lr = LinearRegression()
lr.fit(X_train_level2, y_train_level2)

train_preds = lr.predict(X_train_level2)
r2_train_stacking = r2_score(y_train_level2, train_preds)

test_preds = lr.predict(X_test_level2)
r2_test_stacking = r2_score(y_test, test_preds)

print('Train R-squared for stacking is %f' % r2_train_stacking)
print('Test  R-squared for stacking is %f' % r2_test_stacking)
make_submit(test_preds, "linear_model_mix")
print("END")
print(round((time.time() - time_st) / 60))

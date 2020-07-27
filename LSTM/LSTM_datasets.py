# Adapted from https://www.kaggle.com/sebask/keras-2-0

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
import gc
from torch.utils.data import Dataset, DataLoader


def data_preparation():
    # Import data
    sales = pd.read_csv('data/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
    shops = pd.read_csv('data/shops.csv')
    items = pd.read_csv('data/items.csv')
    cats = pd.read_csv('data/item_categories.csv')
    val = pd.read_csv('data/test.csv')

    # Rearrange the raw data to be monthly sales by item-shop
    df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
    df = df[['date','item_id','shop_id','item_cnt_day']]
    df["item_cnt_day"].clip(0.,20.,inplace=True)
    df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()

    # Merge data from monthly sales to specific item-shops in test data
    test = pd.merge(val,df,on=['item_id','shop_id'], how='left').fillna(0)

    # Strip categorical data so keras only sees raw timeseries
    test = test.drop(labels=['ID','item_id','shop_id'],axis=1)

    # Rearrange the raw data to be monthly average price by item-shop
    # Scale Price
    scaler = MinMaxScaler(feature_range=(0, 1))
    sales["item_price"] = scaler.fit_transform(sales["item_price"].values.reshape(-1,1))
    df2 = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
    df2 = df2[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()

    # Merge data from average prices to specific item-shops in test data
    price = pd.merge(val,df2,on=['item_id','shop_id'], how='left').fillna(0)
    price = price.drop(labels=['ID','item_id','shop_id'],axis=1)

    #########################
    ### mean monthly price of category
    # Fix category
    item_merged_cat = pd.merge(items.drop(['item_name'],axis=1), cats.drop(['item_category_name'],axis=1), how='inner', on='item_category_id')
    sales_cats = pd.merge(sales, item_merged_cat, how='left', on='item_id')
    df3 = sales_cats.groupby([sales_cats.date.apply(lambda x: x.strftime('%Y-%m')), 'item_category_id']).mean().reset_index()
    df3 = df3[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()

    category_price = pd.merge(val,df3,on=['item_id','shop_id'], how='left').fillna(0)
    category_price = category_price.drop(labels=['ID','item_id','shop_id'],axis=1)

    #########################

    # Create x and y training sets from oldest data points
    y_train = test['2015-10']
    x_sales = test.drop(labels=['2015-10'],axis=1)
    x_sales = x_sales.values.reshape((x_sales.shape[0], x_sales.shape[1], 1))
    x_prices = price.drop(labels=['2015-10'],axis=1)
    x_prices= x_prices.values.reshape((x_prices.shape[0], x_prices.shape[1], 1))
    x_category = category_price.drop(labels=['2015-10'],axis=1)
    x_category = x_category.values.reshape((x_category.shape[0], x_category.shape[1], 1))

    X = np.append(x_sales,x_prices,axis=2)
    X = np.append(X, x_category, axis=2)

    y = y_train.values.reshape((214200, 1))
    print("Training Predictor Shape: ",X.shape)
    print("Training Predictee Shape: ",y.shape)
    del y_train, x_sales; gc.collect()

    # Transform test set into numpy matrix
    test = test.drop(labels=['2013-01'],axis=1)
    x_test_sales = test.values.reshape((test.shape[0], test.shape[1], 1))
    x_test_prices = price.drop(labels=['2013-01'],axis=1)
    x_test_prices = x_test_prices.values.reshape((x_test_prices.shape[0], x_test_prices.shape[1], 1))
    
    x_test_category = category_price.drop(labels=['2013-01'],axis=1)
    x_test_category = x_test_category.values.reshape((x_test_category.shape[0], x_test_category.shape[1], 1))

    # Combine Price and Sales Df
    test = np.append(x_test_sales,x_test_prices,axis=2)
    test = np.append(test,x_test_category,axis=2)
    # del x_test_sales,x_test_prices, price; gc.collect()
    print("Test Predictor Shape: ",test.shape)

    return X, y, test


import torch
def to_torch_and_float(x):
    return torch.tensor(x).type(torch.float32)


class train_valid_dataset(Dataset):
    def __init__(self, X, y):
        self.X = to_torch_and_float(X)
        self.Y = to_torch_and_float(y)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index, :], self.Y[index]


class test_dataset(Dataset):
    def __init__(self, X):
        self.X = to_torch_and_float(X)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index, :]

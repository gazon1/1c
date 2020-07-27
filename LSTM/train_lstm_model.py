import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from itertools import product
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
import time
import sys
import gc
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from fastprogress import master_bar, progress_bar
from LSTM_model import LSTM2, LSTM
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler

import copy
import datetime
import random
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# credit goes to https://www.kaggle.com/nicapotato/multivar-lstm-ts-regression-keras

# DONE: добавить цену в датасет
# DONE: реализовать нормирование через StandartScaler
# DONE: уменьшить batch_size до 128
# CANCELED: добавить лаги по таргутеу
# DONE: добавить среднюю цену при group by по категориям в этот месяц - так получу информацию о других магазах в этот месяц
# TODO: Предсказать для валидации моделью xgb с подобранными параметраи 0.9LB и подобрать на валидации коэфициент смешения ответов LSTM, xgb alpha:preds=xgb*alpha+lstm*(1-alpha)

writer = SummaryWriter('runs/training lstm')

test  = pd.read_csv('data/test.csv').set_index('ID')


import random 
import os
SEED = 1345
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)


from torch.utils.data import Dataset, DataLoader
from LSTM_datasets import train_valid_dataset, data_preparation


X, y, test = data_preparation()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=False)
train = train_valid_dataset(X_train, y_train)
valid = train_valid_dataset(X_valid, y_valid)


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))



def train_eval_loop(model, train_dataset, val_dataset, criterion,
                    lr=1e-4, epoch_n=10, batch_size=32,
                    device=None, early_stopping_patience=10, l2_reg_alpha=0,
                    max_batches_per_epoch_train=10000,
                    max_batches_per_epoch_val=1000,
                    data_loader_ctor=DataLoader,
                    optimizer_ctor=None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=0):
    """
    Цикл для обучения модели. После каждой эпохи качество модели оценивается по отложенной выборке.
    :param model: torch.nn.Module - обучаемая модель
    :param train_dataset: torch.utils.data.Dataset - данные для обучения
    :param val_dataset: torch.utils.data.Dataset - данные для оценки качества
    :param criterion: функция потерь для настройки модели
    :param lr: скорость обучения
    :param epoch_n: максимальное количество эпох
    :param batch_size: количество примеров, обрабатываемых моделью за одну итерацию
    :param device: cuda/cpu - устройство, на котором выполнять вычисления
    :param early_stopping_patience: наибольшее количество эпох, в течение которых допускается
        отсутствие улучшения модели, чтобы обучение продолжалось.
    :param l2_reg_alpha: коэффициент L2-регуляризации
    :param max_batches_per_epoch_train: максимальное количество итераций на одну эпоху обучения
    :param max_batches_per_epoch_val: максимальное количество итераций на одну эпоху валидации
    :param data_loader_ctor: функция для создания объекта, преобразующего датасет в батчи
        (по умолчанию torch.utils.data.DataLoader)
    :return: кортеж из двух элементов:
        - среднее значение функции потерь на валидации на лучшей эпохе
        - лучшая модель
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)

    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_workers_n)

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()


            model.train()
            mean_train_loss = 0
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in tqdm(enumerate(train_dataloader), total=round(len(train_dataset)/batch_size)):
                if batch_i > max_batches_per_epoch_train:
                    break

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                # print(batch_x.shape)
                pred = model(batch_x)
                # print(pred.shape, batch_y.shape)
                loss = criterion(pred, batch_y)

                model.zero_grad()
                loss.backward()

                optimizer.step()

                mean_train_loss += float(loss)
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            print('Эпоха: {} итераций, {:0.2f} сек'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))
            print('Среднее значение функции потерь на обучении', mean_train_loss)
            writer.add_scalar('training loss',
                            mean_train_loss,
                            epoch_i)



            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in tqdm(enumerate(val_dataloader), total=round(len(val_dataset)/batch_size)):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)

                    mean_val_loss += float(loss)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print('Среднее значение функции потерь на валидации', mean_val_loss)
            writer.add_scalar('valid loss',
                            mean_train_loss,
                            epoch_i)

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print('Новая лучшая модель!')
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                    early_stopping_patience))
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
        except Exception as ex:
            print('Ошибка при обучении: {}\n{}'.format(ex, traceback.format_exc()))
            break

    return best_val_loss, best_model


#####  Parameters  ######################
from LSTM_model import get_params
params = get_params()
num_epochs = params["num_epochs"]
learning_rate = params["learning_rate"]
batch_size = params["batch_size"]
input_size = params["input_size"]
hidden_size = params["hidden_size"]
num_layers = params["num_layers"]
num_classes = params["num_classes"]

#####Init the Model #######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
lstm = LSTM(num_classes, input_size, hidden_size, num_layers, device).to(device)


##### Set Criterion Optimzer and scheduler ####################
criterion = torch.nn.MSELoss().to(device)   # mean-squared error for regression

best_val_loss, best_model = \
train_eval_loop(lstm,
                train,
                valid,
                criterion,
                lr=learning_rate,
                epoch_n = num_epochs,
                batch_size=batch_size,
                device=device,
                l2_reg_alpha=1e-5,
                shuffle_train=True,
                dataloader_workers_n=1,
                lr_scheduler_ctor = lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08),
                max_batches_per_epoch_train = round(6186922 / batch_size - 1),
                max_batches_per_epoch_val = round(238172 / batch_size - 1))

torch.save(best_model.state_dict(), "./best_LSTM_model.pth")
print(f"{best_val_loss} - best val loss")


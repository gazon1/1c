import pandas as pd
from LSTM_model import LSTM
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
from LSTM_datasets import test_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

lstm = LSTM(num_classes, input_size, hidden_size, num_layers, device).to(device)
lstm.load_state_dict(torch.load("best_LSTM_model.pth"))
lstm.eval()

from LSTM_datasets import data_preparation
_, _, test = data_preparation()
test = test_dataset(test)

preds = []
from tqdm import tqdm
test_dataloader = DataLoader(test, batch_size=2048, shuffle=False,
                                      num_workers=1)
for x in tqdm(test_dataloader):
    _x = lstm(x).detach().numpy()
    shape = _x.shape
    _x = _x.reshape(shape[0], shape[1])
    preds.append(_x)

pred = np.row_stack(preds)
pred = np.clip(pred, 0, 20) # Clip predictions to suit test target distribution

submission = pd.DataFrame(pred,columns=['item_cnt_month'])
out_file = 'submission_lstm.csv'
submission.to_csv(out_file,index_label='ID')
print(f"Predictions at {out_file}")

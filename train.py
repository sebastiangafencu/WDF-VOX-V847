import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from model import BJTModel

class BJTDataset(Dataset):
    def __init__(self, z: list, df, root: str = 'data', device: str = 'cuda'):
        super().__init__()
        self.root = root
        self.data_df = df
        self.z = z
        self.device = device
        self.load_data()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index][0:2]
        y = self.data[index][2:4]
        return x, y

    def load_data(self):
        v1 = torch.Tensor(self.data_df['Vbe'].to_numpy())
        v2 = torch.Tensor(self.data_df['Vbc'].to_numpy())
        i1 = torch.Tensor(self.data_df['Ibe'].to_numpy())
        i2 = torch.Tensor(self.data_df['Ibc'].to_numpy())

        self.data = torch.zeros([v1.shape[0], 4], device=self.device, dtype=torch.float64)
        self.data[:, 0] = v1
        self.data[:, 1] = v2
        self.data[:, 2] = i1
        self.data[:, 3] = i2

def k2w(z, v1, v2, i1, i2, v3, v4, i3, i4):
    a = torch.zeros([v1.shape[0], 4], device='cuda')
    b = torch.zeros([v1.shape[0], 4], device='cuda')

    a[:, 0] = v1 + z[0] * i1 + z[4] * i2 + z[8] * i3 + z[12] * i4
    a[:, 1] = v2 + z[1] * i1 + z[5] * i2 + z[9] * i3 + z[13] * i4
    a[:, 2] = v3 + z[2] * i1 + z[6] * i2 + z[10] * i3 + z[14] * i4
    a[:, 3] = v4 + z[3] * i1 + z[7] * i2 + z[11] * i3 + z[15] * i4
    b[:, 0] = v1 - z[0] * i1 - z[4] * i2 - z[8] * i3 - z[12] * i4
    b[:, 1] = v2 - z[1] * i1 - z[5] * i2 - z[9] * i3 - z[13] * i4
    b[:, 2] = v3 - z[2] * i1 - z[6] * i2 - z[10] * i3 - z[14] * i4
    b[:, 3] = v4 - z[3] * i1 - z[7] * i2 - z[11] * i3 - z[15] * i4

    return a, b

def w2v(x, y_pred):
    return (y_pred+x)/2

def w2i(x, y_pred, z_inv):
    return 0.5*torch.matmul(x-y_pred, z_inv)

def train_loop(dataloader1, dataloader2, model, loss_fn, optimizer, device, epoch):
    size = len(dataloader1.dataset)
    batch_size = dataloader1.batch_size
    pbar = tqdm(total=size, ascii=True, leave=True, position=0)
    pbar.set_description(f"Epoch {epoch + 1} training")

    model.train()
    dataloader2_iterator = iter(dataloader2)
    for v12, i12 in dataloader1:
        v34, i34 = next(dataloader2_iterator)
        X, y = k2w(z, v12[:, 0], v12[:, 1], i12[:, 0], i12[:, 1], v34[:, 0], v34[:, 1], i34[:, 0], i34[:, 1])

        # Compute prediction and loss
        pred = model(X)
        if kirchhoff_domain_loss:
            v = torch.zeros([X.shape[0], 4], device='cuda')
            v[:, 0:2] = v12
            v[:, 2:4] = v34
            i = torch.zeros([X.shape[0], 4], device='cuda')
            i[:, 0:2] = i12
            i[:, 2:4] = i34

            pred_v = w2v(X, pred)
            pred_i = w2i(X, pred, z_inv)
            if weighted_loss:
                loss = 2*loss_fn(pred_v[:, 0], v[:, 0]) + loss_fn(pred_v[:, 1], v[:, 1]) + 2*loss_fn(pred_v[:, 2],v[:,2]) + loss_fn(pred_v[:, 3], v[:, 3])
            else:
                loss = loss_fn(pred_v, v) + loss_fn(pred_i, i)
        else:
            if weighted_loss:
                loss = 2 * loss_fn(pred[:, 0], y[:, 0]) + loss_fn(pred[:, 1], y[:, 1]) + 2 * loss_fn(pred[:, 2],y[:, 2]) + loss_fn(pred[:, 3], y[:, 3])
            else:
                loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update(batch_size)
    pbar.close()


def test_loop(dataloader1, dataloader2, model, loss_fn, device, epoch):
    size = len(dataloader1.dataset)
    batch_size = dataloader1.batch_size
    num_batches = len(dataloader1)
    test_loss = 0
    pbar = tqdm(total=size, ascii=True, leave=True, position=0)
    pbar.set_description(f"Epoch {epoch + 1} testing")
    model.eval()
    dataloader2_iterator = iter(dataloader2)
    with torch.no_grad():
        for v12, i12 in dataloader1:
            v34, i34 = next(dataloader2_iterator)
            X, y = k2w(z, v12[:, 0], v12[:, 1], i12[:, 0], i12[:, 1], v34[:, 0], v34[:, 1], i34[:, 0], i34[:, 1])
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pbar.update(batch_size)
    test_loss /= num_batches
    pbar.close()
    print(f"Avg loss: {test_loss} \n")
    return test_loss

def nmse_loss(yhat, y):
    return torch.sum((yhat - y) ** 2) / torch.sum(y ** 2)

def cauchy_loss(yhat, y):
    c = 0.05
    error = yhat - y
    loss = torch.log(1 + (error / c) ** 2)
    return torch.mean(loss)

def split_dataset(df):
    np.random.seed(0)
    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= 0.9
    train = df[msk]
    test = df[~msk]
    train = train.drop('split', axis=1)
    test = test.drop('split', axis=1)
    return train, test

torch.set_default_dtype(torch.float64);
device = 'cuda'

# Load dataset, split into train and test set
data = pd.read_csv('data/dataset_v13.csv', sep=',', header=0)
data_bjt1 = data[['Vbe1', 'Vbc1', 'Ibe1', 'Ibc1']]
data_bjt2 = data[['Vbe2', 'Vbc2', 'Ibe2', 'Ibc2']]
data_bjt1 = data_bjt1.rename(columns={'Vbe1': 'Vbe', 'Vbc1': 'Vbc', 'Ibe1': 'Ibe', 'Ibc1': 'Ibc'})
data_bjt2 = data_bjt2.rename(columns={'Vbe2': 'Vbe', 'Vbc2': 'Vbc', 'Ibe2': 'Ibe', 'Ibc2': 'Ibc'})
train_bjt1, test_bjt1 = split_dataset(data_bjt1)
train_bjt2, test_bjt2 = split_dataset(data_bjt2)

batch_size = 64

z = torch.tensor([7.582196517735750e+03, 7.072121313302530e+03, -4.665240435820493e+03, 0.039470183442808,
     7.072121313302530e+03, 2.399945437144726e+04, -1.354933372697417e+04, -8.884116433041805e+03,
     -4.665240435820493e+03, -1.354933372697417e+04, 3.424869882868201e+04, 2.851618556414040e+04,
     0.039470183442808, -8.884116433041805e+03, 2.851618556414040e+04, 2.951621842536739e+04])
z_inv = torch.inverse(torch.reshape(z.to(device), (4,4)))

train_data_bjt1 = BJTDataset(z=z, df=train_bjt1, device=device)
train_dataloader_bjt1 = DataLoader(train_data_bjt1, batch_size=batch_size, shuffle=True)
test_data_bjt1 = BJTDataset(z=z, df=test_bjt1, device=device)
test_dataloader_bjt1 = DataLoader(test_data_bjt1, batch_size=batch_size, shuffle=True)

train_data_bjt2 = BJTDataset(z=z, df=train_bjt2, device=device)
train_dataloader_bjt2 = DataLoader(train_data_bjt2, batch_size=batch_size, shuffle=True)
test_data_bjt2 = BJTDataset(z=z, df=test_bjt2, device=device)
test_dataloader_bjt2 = DataLoader(test_data_bjt2, batch_size=batch_size, shuffle=True)

hidden_dim = 64
num_layers = 4
kirchhoff_domain_loss = True
weighted_loss = True
activation = "elu"
model = BJTModel(hidden_dim=hidden_dim, num_layers=num_layers, device=device, input_size=4, output_size=4, activation=activation)

epochs = 500
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loss_fn = cauchy_loss
valid_loss_fn = nmse_loss

model_desc = "".join(
            [str('tp5_dataset13_cauchy=0.05'), '_kDom=',str(kirchhoff_domain_loss),  '_activation=', str(activation), '_bS=', 
             str(batch_size), '_hDim=',  str(hidden_dim),
             '_nLayers=', str(num_layers)])

best_model = [model.state_dict(), '', 5e-07]
train_history = []
for t in range(epochs):
    train_loop(train_dataloader_bjt1, train_dataloader_bjt2, model, train_loss_fn, optimizer, device, t)
    test_loss = test_loop(test_dataloader_bjt1, test_dataloader_bjt2, model, valid_loss_fn, device, t)

    if (test_loss < best_model[2]):
        best_model = [model.state_dict(), "".join(
            ['checkpoints/', model_desc, '_lr=', str(learning_rate), '_epoch=',str(t),'_trainLoss=', str(test_loss), '.pth']), test_loss]
        print("Checkpoint reached, test loss: ", best_model[1])
        torch.save(best_model[0], best_model[1])

    train_history.append(test_loss)
    plt.figure()
    plt.plot(range(t + 1), train_history, label='train loss')
    plt.grid()
    plt.yscale('log')
    plt.title(model_desc, fontsize=6)
    plt.savefig("".join(['plots/',model_desc,'.png']))
    plt.close()

# import
import pandas as pd
from os.path import join, isfile
from glob import glob
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from model import *
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import requests

#global parameters
USE_CUDA = torch.cuda.is_available()

# def


def train_loop(dataloader, model, optimizer, criterion, epochs):
    train_loader, test_loader = dataloader
    history = []
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = []
        for x, y in train_loader:
            if USE_CUDA:
                x, y = x.cuda(), y.cuda()
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        history.append(np.mean(total_loss))
    return history


def evaluation(dataloader, model):
    predict = []
    accuracy = []
    model.eval()
    for x, y in dataloader:
        if USE_CUDA:
            x, y = x.cuda(), y.cuda()
        pred = model(x).cpu().data.numpy()
        pred = np.argmax(pred, 1)
        acc = accuracy_score(y.cpu().data.numpy(), pred)
        predict.append(pred)
        accuracy.append(acc)
    return predict, accuracy


if __name__ == "__main__":
    # parameters
    data_path = './data'
    batch_size = 1000
    lr = 0.0001
    epochs = 20
    url = 'https://doc-14-80-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/q8k52e3qfie8lk20d225dogn90t2sdue/1580428800000/07963431181216383630/*/1AasIiFr9zAZr2_zGwONb5pQIMhWOoBSx?e=download'
    filename = 'A_Z_Handwritten_Data.csv'

    # download data
    if not isfile(join(data_path, filename)):
        print('download file from url')
        r = requests.get(url, stream=True)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(join(data_path, filename), 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

    # load data
    print('load data')
    df = pd.read_csv(glob(join(data_path, '*.csv'))[-1])
    data = df.values[:, 1:]
    label = df.values[:, 0].astype(int)

    # preprocessing
    x_train, x_test, y_train, y_test = train_test_split(
        data, label, test_size=0.3)
    train_set = TensorDataset(torch.from_numpy(
        x_train).float(), torch.from_numpy(y_train))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_set = TensorDataset(torch.from_numpy(
        x_test).float(), torch.from_numpy(y_test))
    test_loader = DataLoader(dataset=test_set, num_workers=4, pin_memory=True)

    # create model
    print('create model')
    mlp = MLP(in_dim=784, out_dim=26, hidden_dim=256,
              n_hidden=1, dropout=0.5)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    if USE_CUDA:
        mlp = mlp.cuda()

    # train
    print('training')
    history = train_loop((train_loader, test_loader), mlp,
                         optimizer, criterion, epochs)

    # evaluation
    print('evaluation')
    train_pred, train_acc = evaluation(train_loader, mlp)
    print('Train dataset accuracy: {}'.format(round(np.mean(train_acc), 4)))
    test_pred, test_acc = evaluation(test_loader, mlp)
    print('Test dataset accuracy: {}'.format(round(np.mean(test_acc), 4)))

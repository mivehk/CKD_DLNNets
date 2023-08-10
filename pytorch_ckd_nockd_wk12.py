!pip install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

'''
Looking in links: https://download.pytorch.org/whl/torch_stable.html
Collecting torch==1.13.1+cpu
  Downloading https://download.pytorch.org/whl/cpu/torch-1.13.1%2Bcpu-cp37-cp37m-linux_x86_64.whl (199.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 199.1/199.1 MB 3.2 MB/s eta 0:00:0000:0100:01
Requirement already satisfied: typing-extensions in /opt/conda/lib/python3.7/site-packages (from torch==1.13.1+cpu) (4.1.1)
Installing collected packages: torch
Successfully installed torch-1.13.1+cpu

[notice] A new release of pip available: 22.3.1 -> 23.2.1
[notice] To update, run: pip install --upgrade pip
'''

!pip show torch

'''
Name: torch
Version: 1.13.1+cpu
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /home/jupyter/.local/lib/python3.7/site-packages
Requires: typing-extensions
Required-by: 
'''

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

##import 'torch_nockd_ckd.csv' from orkspace bucket
from sklearn.metrics import roc_auc_score

import torch
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # input observations with their target class present the problem of interest
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
 
# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 6)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(6, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X
 
# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=512, shuffle=False)
    return train_dl, test_dl


 
# train the model
def train_model(train_dl, model):
    # define the optimization (e.g., backpropagate calculate error gredients of loss function with respect to weight of the network)
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches of 100 epoch with train_data of 8x[32,34] and 8x[32,1] to satisfy train data size of close to 256 rows
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            ## gredient vector is a vector of partial derivatives of function f with respect to(w.r.t) each independent variables.
            ## partial derivative is rate of change for function f with respoect to variable x. denoted as "f w.r.t x" which is ∂f/∂x 
            ##https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors  
            yhat = model(inputs)
            # calculate loss for model output -
            # gredients are derivative of the loss function showing how loss changes in respect to each parameter  
            # if pytorch.no_grad() is used then these gredients are not computed like during testing when we do not backpropagate.
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
 
# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc
 
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

#path= 'torch_nockd.csv' dataset asked by dr. yin in wk13
path= 'torch_nockd_ckd.csv' #dataset used
train_dl, test_dl = prepare_data(path)
print( len(train_dl.dataset), len(test_dl.dataset))
model = MLP(6)
train_model(train_dl, model)
 

predictions1, actuals1 = list(), list()
for i, (inputs, targets) in enumerate(test_dl):
    # evaluate the model on the test set
    yhat1 = model(inputs)
    # retrieve numpy array
    yhat1 = yhat1.detach().numpy()
    actual1 = targets.numpy()
    actual1 = actual1.reshape((len(actual1), 1))
    # round to class values
    yhat1 = yhat1.round()
    # store
    predictions1.append(yhat1)
    actuals1.append(actual1)
predictions1, actuals1 = vstack(predictions1), vstack(actuals1)
# calculate accuracy
print(len(predictions1)) ##749
print(len(actuals1))  #749


auc = roc_auc_score(predictions1, actuals1)
print("AUC:", auc) #AUC: 0.7378453966584881


acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc) #Accuracy: 0.722

com = confusion_matrix( predictions1, actuals1)
sns.heatmap(com, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


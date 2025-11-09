import os
import pandas as pd
from pandas import read_csv
import numpy as np
from numpy import vstack

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Sigmoid, Module, BCELoss
from torch.optim import SGD
from torch.nn.init import kaiming_uniform_, xavier_uniform_

## from torch.nn.parallel import DistributedDataParallel as DDP
## from torch.utils.data.distributed import DistributedSampler

name_of_file_in_bucket = 'torch_nockd_ckd.csv'
my_bucket = os.getenv('WORKSPACE_BUCKET')
os.system(f"gsutil cp '{my_bucket}/data/{name_of_file_in_bucket}' .")
print(f'[INFO] {name_of_file_in_bucket} is successfully downloaded into your working space')

class CSVDataset(Dataset):
  
    def __init__(self, path):
        # load the csv file into memory as a pandas dataframe
        df = read_csv(path)
        continuous_cols = ['egfr', 'hba1c']
        categorical_cols = ['sab', 'race_black', 'race_asian', 'race_other']
        Scaler = StandardScaler()
        df[continuous_cols] = Scaler.fit_transform(df[continuous_cols])
        # store the inputs and outputs into numpy arrays - DataLoader will convert to tensor before passing to model
        # self.X = df.values[:, :-1]
        self.X = df[continuous_cols + categorical_cols].values
        self.y = df.values[:, -1]
        # input observations with their target class present the problem of interest
        # ensure input data to pytorch are floating-point numbers.
        # self.X.shape[0] is number of rows and self.X.shape[1] is number of features
        self.X = self.X.astype('float32')
        # learn the mapping of labels into indices and transform them.
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        # convert one dimensional labels into column vector (2D) - numpy shape gets tuple even if it is 1d like (7,)
        self.y = self.y.reshape((len(self.y), 1))
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at a certain index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    # split dataset into random list of indices defined by list items
    # No generator means indices objects are non-deterministic without manual_seed
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # function returns subset objects of torch dataset holding random indices, so every run gets random rows 
        return random_split(self, [train_size, test_size])
 

class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        ##self.hidden1 = Linear(n_inputs, 10).to('cuda:0')
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 6)
        ##self.hidden2 = Linear(10, 6).to('cuda:1')
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
    dataset = CSVDataset(path)
    # subset objects of dataset containing random indices
    train, test = dataset.get_splits()
    ## distribute for bigdata
    ##train_sampler = DistributedSampler(train_dataset)
    # dataloader objects that become tensor before getting passed into model
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    ##train_dl = DataLoader(train, sampler=train_sampler, batch_size=32)
    test_dl = DataLoader(test, batch_size=512, shuffle=False)
    return train_dl, test_dl


 
# train the model
def train_model(train_dl, model):
    # loss is the error between predicted probability and actual values
    criterion = BCELoss()
    # backpropagation computes gradients of the loss function with respect to every weight and bias
    # then optimizer uses those gradients with learning rate to update weights 
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # each epoch is one through pass on dataset
    for epoch in range(100):
        # enumerate mini batches of tensors generated by dataloader object 8x[32,6] and 8x[32,1] which is set of 256 divided by8 batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the previous gradients
            optimizer.zero_grad()
            # compute the model output by running inputs through hidden/activation layers
            ## gredient vector is a vector of partial derivatives of function f with respect to(w.r.t) each independent variables.
            ## partial derivative is rate of change for function f with respoect to variable x. denoted as "f w.r.t x" which is ∂f/∂x 
            ## https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors  
            yhat = model(inputs)
            # calculate loss for model output -
            # gredients are derivative of the loss function showing how loss changes in respect to each parameter  
            # if pytorch.no_grad() is used then these gredients are not computed like during testing when we do not backpropagate.
            loss = criterion(yhat, targets)
            # calculate gradients of loss with respect to all model parameters
            loss.backward()
            # update model weights
            optimizer.step()
 
# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    # iterate over dataloader object, which provides batches of tensors
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve predicted probabilities and ground-truth labels by numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round predicted values 
        # because ValueError: Classification metrics can't handle a mix of binary and continuous targets (probabilities)
        yhat = yhat.round()
        # store batch predictions and labels
        predictions.append(yhat)
        actuals.append(actual)    
    predictions, actuals = vstack(predictions), vstack(actuals) # stack all batches vertically into full test-set arrays
    acc = accuracy_score(actuals, predictions) # first parameter must be labels
    auc = roc_auc_score(actuals, predictions)
    return acc, auc, predictions, actuals
 
# make a class prediction for one row of data
def predict(row, model):
    # wrap one row of features from numpy array shape of (6,) within list to become 2d and accepteed by Tensor
    row = Tensor([row])
    # model expects batch dimension (2d) to make prediction on that Tensor with shape of torch.Size([1, 6])
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


name_of_file_in_bucket = 'torch_nockd_ckd.csv'
path = name_of_file_in_bucket # 'torch_nockd_ckd.csv' #The name of used dataset
train_dl, test_dl = prepare_data(path)
print( len(train_dl.dataset), len(test_dl.dataset)) # 1521 749
model = MLP(6)
## model = DDP(model)
train_model(train_dl, model)
acc, auc, preds, labels = evaluate_model(test_dl, model)


print("AUC:", auc) 
#AUC: 0.8422897196261683
print('Accuracy: %.3f' % acc) 
#Accuracy: 0.853

com = confusion_matrix(labels, preds)
sns.heatmap(com, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


"""
Functions for setting up the pytorch dataset and model
Author: Brianna Richardson

Script utilized for FAccT '23 paper:



"""

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.autograd import grad
from torch.autograd.functional import hessian as hs
from torch.nn.utils import _stateless
import numpy as np
import pandas as pd

class CurrentDataset(Dataset):
    def __init__(self, X, Y, W=None):
        X = X.copy()
        self.X = X
        self.Y = Y
        if W is None:
            self.W = [1]*len(Y)
        else:
            self.W = W.tolist()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.W[idx]
    
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
    def forward(self, x, temp=1):
        x = torch.sigmoid(torch.div(self.linear(x.float()),temp))
        return x
    
def train(trainloader, model):
    epochs = 150
    learning_rate = 0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(int(epochs)):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels, w = data[0].cpu(), data[1].cpu(), data[2]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs,1)
            labels = labels.unsqueeze(1)
            labels = labels.float()

            # weighted loss
            def weighted_loss(y, y_hat, w):
                return (criterion(y, y_hat)*w).mean()

            loss = weighted_loss(outputs, labels, w)

            loss.backward()
            optimizer.step()
            
def save_model(model, PATH):
    torch.save(model.state_dict(), PATH)


def load_model(input_dim, PATH):
    model = LogisticRegression(input_dim)
    model.load_state_dict(torch.load(PATH))
    model.cpu()
    return model


def test(testloader, model):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            attr, labels = data[0].cpu(), data[1].cpu()
            outputs = model(attr,1)
            #predicted = torch.gt(outputs.data, 0.5).type(torch.FloatTensor)
            total += labels.size(0)
            correct += torch.sum(outputs.squeeze() == labels).item()

    print('Accuracy of the model on the test set: %d %%' % (100*correct/total))

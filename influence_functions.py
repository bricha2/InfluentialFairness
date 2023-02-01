# This file runs the model
import sys
sys.path.insert(0, "/blue/juangilbert/richardsonb/influence_functions/ctfdist-master/")
sys.path.insert(0, "/blue/juangilbert/richardsonb/influence_functions/ctfdist-master/scripts")
sys.path.append('../')
from rankedlist import *

# For Wang paper
import ctfdist
from ctfdist.paths import *
from ctfdist.descent import *
from ctfdist.helper_functions import *
from plot_helper import *

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.autograd import grad
from torch.autograd.functional import hessian as hs
from torch.nn.utils import _stateless
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score,make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split, GridSearchCV

from pathlib import Path
from os.path import exists
from PIL import Image
    
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
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in tqdm(range(int(epochs)), desc='Training Epochs'):
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

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    """#for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):  # loop over the dataset multiple times
    for epoch in range(int(epochs)):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data
            inputs, labels = data[0].cpu(), data[1].cpu()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs,1)
            labels = labels.unsqueeze(1)
            labels=labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0'"""

    #print('Finished Training')

# From CTFdist - Wang et al
def df_to_XYS(df, outcome_column_name = None, group_column_name = None, minority_group = 0):
    if outcome_column_name is not None:
        Y = df[outcome_column_name].values
    else:
        Y = df.iloc[:, 0].values

    if group_column_name is not None:
        G = df[group_column_name].values
    else:
        G = df.iloc[:, 1].values

    S = np.isin(G, minority_group, invert = True)
    X = np.array(df.iloc[:, 2:].values, dtype = np.float64)
    Y = np.array(Y, dtype = np.float64)
    return X, Y, S

# From CTFdist - Wang et al
def train_test_audit_split(df, train_size, test_size, audit_size, outcome_column_name, group_column_name, random_state):

    # split into audit and other
    df_audit, df_other = train_test_split(df,
                                          train_size = audit_size,
                                          stratify = df[[group_column_name, outcome_column_name]],
                                          random_state = random_state)

    # split other into train and test
    other_size = train_size + test_size
    train_size = train_size / other_size
    test_size = test_size / other_size
    df_train, df_test = train_test_split(df_other,
                                         train_size = train_size,
                                         stratify = df_other[[group_column_name, outcome_column_name]],
                                         random_state = random_state)
    return df_train, df_audit, df_test


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

def grad_individual(attr, label, model, loss_fxn):
    """Calculates the gradient z for one sample."""
    model.eval()
    attr, label = attr.cpu(), torch.unsqueeze(label.float(),0).cpu()
    y = model(attr)
    loss = loss_fxn(y, label)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    return grad(loss, params, create_graph=True)

def grad_group(attr, labels, model, loss_fxn):
    """Calculates the gradient z for several samples."""
    model.eval()
    labels = labels.unsqueeze(1).float()
    y = model(attr)
    loss = loss_fxn(y, labels)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    grad_loss = grad(loss, params, create_graph=True)
    grad_loss = torch.vstack([g.view(-1,1) for g in grad_loss])
    return grad_loss

def hessian(grad_loss,model):
    """Calculates the hessian."""
    loss_hessian = []
    for i in range(grad_loss.size(0)):
        hs = grad(grad_loss[i], model.parameters(), retain_graph=True)
        hs = torch.vstack([h.view(-1,1) for h in hs])
        loss_hessian.append(hs)
    return torch.hstack(loss_hessian)

def calc_disp(model, attr, labels, sensitive, sensitive_groups = ['0','1']):
    # calculate predicted
    model.eval()
    predicted = model(attr,0.0001)
    predicted = predicted.squeeze()

    # Make a table of results
    scores = pd.DataFrame()

    # Add shape to table
    scores['n_pos'] = [torch.sum(labels).detach().numpy(), torch.sum(labels[sensitive==0]).detach().numpy(), torch.sum(labels[sensitive==1]).detach().numpy()]
    scores['n_neg'] = [torch.sum(torch.add(labels[labels==0],1)).detach().numpy(), torch.sum(torch.add(labels[(sensitive==0)*(labels==0)],1)).detach().numpy(), torch.sum(torch.add(labels[(sensitive==1)*(labels==0)],1)).detach().numpy()]
    scores['n_ppos'] = [torch.sum(predicted).detach().numpy(), torch.sum(predicted[sensitive==0]).detach().numpy(), torch.sum(predicted[sensitive==1]).detach().numpy()]
    scores['n_pneg'] = [torch.sum(torch.add(predicted[predicted==0],1)).detach().numpy(), torch.sum(torch.add(predicted[(sensitive==0)*(predicted==0)],1)).detach().numpy(), torch.sum(torch.add(predicted[(sensitive==1)*(predicted==0)],1)).detach().numpy()]

    # Add row names
    scores.index = ['All', sensitive_groups[0],sensitive_groups[1]]

    # calculate the tp/fn/tn/fp for each group and total
    tp = torch.sum(predicted[(labels==1)])
    fp = torch.sum(predicted[(labels==0)])
    tn = torch.sum(torch.add(predicted[(predicted.squeeze()==0)*(labels==0)],1))
    fn = torch.sum(torch.add(predicted[(predicted.squeeze()==0)*(labels==1)],1))
    true = torch.sum(torch.add(tp, tn))
    false = torch.sum(torch.add(fp, fn))
    total = torch.sum(torch.add(true, false))
    tp_sens0 = torch.sum(predicted[(sensitive==0)*(labels==1)])
    tp_sens1 = torch.sum(predicted[(sensitive==1)*(labels==1)])
    fn_sens0 = torch.sum(torch.add(predicted[(sensitive==0)*(predicted.squeeze()==0)*(labels==1)],1))
    fn_sens1 = torch.sum(torch.add(predicted[(sensitive==1)*(predicted.squeeze()==0)*(labels==1)],1))
    fp_sens0 = torch.sum(predicted[(sensitive==0)*(labels==0)])
    fp_sens1 = torch.sum(predicted[(sensitive==1)*(labels==0)])
    tn_sens0 = torch.sum(torch.add(predicted[(sensitive==0)*(predicted.squeeze()==0)*(labels==0)],1))
    tn_sens1 = torch.sum(torch.add(predicted[(sensitive==1)*(predicted.squeeze()==0)*(labels==0)],1))
    true_sens0 = torch.sum(torch.add(tp_sens0, tn_sens0))
    false_sens0 = torch.sum(torch.add(fp_sens0, fn_sens0))
    total_sens0 = torch.sum(torch.add(true_sens0, false_sens0))
    true_sens1 = torch.sum(torch.add(tp_sens1, tn_sens1))
    false_sens1 = torch.sum(torch.add(fp_sens1, fn_sens1))
    total_sens1 = torch.sum(torch.add(true_sens1, false_sens1))

    # Add metrics
    scores['TPR'] = [torch.div(tp,torch.add(tp,fn)).detach().numpy(),torch.div(tp_sens0,torch.add(tp_sens0,fn_sens0)).detach().numpy(),torch.div(tp_sens1,torch.add(tp_sens1,fn_sens1)).detach().numpy()]
    scores['TNR'] = [torch.div(tn,torch.add(tn,fp)).detach().numpy(),torch.div(tn_sens0,torch.add(tn_sens0,fp_sens0)).detach().numpy(),torch.div(tn_sens1,torch.add(tn_sens1,fp_sens1)).detach().numpy()]
    scores['FPR'] = [torch.div(fp,torch.add(fp,tn)).detach().numpy(),torch.div(fp_sens0,torch.add(fp_sens0,tn_sens0)).detach().numpy(),torch.div(fp_sens1,torch.add(fp_sens1,tn_sens1)).detach().numpy()]
    scores['FNR'] = [torch.div(fn,torch.add(fn,tp)).detach().numpy(),torch.div(fn_sens0,torch.add(fn_sens0,tp_sens0)).detach().numpy(),torch.div(fn_sens1,torch.add(fn_sens1,tp_sens1)).detach().numpy()]
    scores['FDR'] = [torch.div(fp,torch.add(fp,tp)).detach().numpy(),torch.div(fp_sens0,torch.add(fp_sens0,tp_sens0)).detach().numpy(),torch.div(fp_sens1,torch.add(fp_sens1,tp_sens1)).detach().numpy()]
    scores['FOR'] = [torch.div(fn,torch.add(fn,tn)).detach().numpy(),torch.div(fn_sens0,torch.add(fn_sens0,tn_sens0)).detach().numpy(),torch.div(fn_sens1,torch.add(fn_sens1,tn_sens1)).detach().numpy()]
    scores['SP'] = [torch.mean(predicted).detach().numpy(), torch.mean(predicted[(sensitive==0)]).detach().numpy(), torch.mean(predicted[(sensitive==1)]).detach().numpy()]
    scores['EO'] = [torch.mean(predicted[(labels==1)]).detach().numpy(), torch.mean(predicted[(sensitive==0)*(labels==1)]).detach().numpy(), torch.mean(predicted[(sensitive==1)*(labels==1)]).detach().numpy()]
    scores['ACC'] = [torch.div(true, total).detach().numpy(), torch.div(true_sens0, total_sens0).detach().numpy(), torch.div(true_sens1, total_sens1).detach().numpy()]
    return scores

def calc_disp_nomodel(labels, preds, sensitive, sensitive_groups = ['0','1']):

    # Make a table of results
    scores = pd.DataFrame()

    scores['n_pos'] = [np.sum(labels), np.sum(labels[sensitive==0]), np.sum(labels[sensitive==1])]
    scores['n_neg'] = [np.sum(labels == 0), np.sum(labels[sensitive==0] == 0), np.sum(labels[sensitive==1] == 0)]
    scores['n_ppos'] = [np.sum(preds), np.sum(preds[sensitive==0]), np.sum(preds[sensitive==1])]
    scores['n_pneg'] = [np.sum(preds == 0), np.sum(preds[sensitive==0] == 0), np.sum(preds[sensitive==1] == 0)]

    # Add row names
    scores.index = ['All', sensitive_groups[0],sensitive_groups[1]]

    # calculate the tp/fn/tn/fp for each group and total
    tp = np.sum(preds[(labels==1)])
    fp = np.sum(preds[(labels==0)])
    tn = np.sum(preds[labels == 0] == 0)
    fn = np.sum(preds[labels == 1] == 0)
    true = tp + tn
    false = fp + fn
    total = true + false
    tp_sens0 = np.sum(preds[(labels==1)*(sensitive==0)])
    tp_sens1 = np.sum(preds[(labels==1)*(sensitive==1)])
    fn_sens0 = np.sum(preds[(labels==1)*(sensitive==0)]==0)
    fn_sens1 = np.sum(preds[(labels==1)*(sensitive==1)]==0)
    fp_sens0 = np.sum(preds[(labels==0)*(sensitive==0)])
    fp_sens1 = np.sum(preds[(labels==0)*(sensitive==1)])
    tn_sens0 = np.sum(preds[(labels==0)*(sensitive==0)]==0)
    tn_sens1 = np.sum(preds[(labels==0)*(sensitive==1)]==0)
    true_sens0 = tp_sens0 + tn_sens0
    false_sens0 = fp_sens0 + fn_sens0
    total_sens0 = true_sens0 + false_sens0
    true_sens1 = tp_sens1 + tn_sens1
    false_sens1 = fp_sens1 + fn_sens1
    total_sens1 = true_sens1 + false_sens1

    # Add metrics
    scores['TPR'] = [tp/(tp + fn), tp_sens0/(tp_sens0 + fn_sens0), tp_sens1/(tp_sens1 + fn_sens1)]
    scores['TNR'] = [tn/(tn + fp), tn_sens0/(tn_sens0 + fp_sens0), tn_sens1/(tn_sens1 + fp_sens1)]
    scores['FPR'] = [fp/(fp + tn), fp_sens0/(fp_sens0 + tn_sens0), fp_sens1/(fp_sens1 + tn_sens1)]
    scores['FNR'] = [fn/(fn + tp), fn_sens0/(fn_sens0 + tp_sens0), fn_sens1/(fn_sens1 + tp_sens1)]
    scores['FDR'] = [fp/(fp + tp), fp_sens0/(fp_sens0 + tp_sens0), fp_sens1/(fp_sens1 + tp_sens1)]
    scores['FOR'] = [fn/(fn + tn), fn_sens0/(fn_sens0 + tn_sens0), fn_sens1/(fn_sens1 + tn_sens1)]
    scores['SP'] = [np.mean(preds), np.mean(preds[sensitive==0]), np.mean(preds[sensitive==1])]
    scores['EO'] = [np.mean(preds[(labels==1)]),np.mean(preds[(labels==1)*(sensitive==0)]),np.mean(preds[(labels==1)*(sensitive==1)])]
    scores['ACC'] = [true/total, true_sens0/total_sens0, true_sens1/total_sens1]

    return scores

def grad_disp(model, metric, attr, labels, sensitive, protected_group=0, fav_label=1):
    # Swap fav_label if its the other way around
    if fav_label == 0:
        print("Favored label should be 1")
        fav_label = 1
    
    # calculate predicted
    model.eval()
    predicted = model(attr,1)
    #outputs = model(attr)

    if metric == 'TPR':
        tp_sens0 = torch.sum(predicted[(sensitive==protected_group)*(labels==fav_label)])
        tp_sens1 = torch.sum(predicted[(sensitive!=protected_group)*(labels==fav_label)])
        fn_sens0 = torch.sum(torch.add(predicted[(sensitive==protected_group)*(predicted.squeeze()!=fav_label)*(labels==fav_label)],1))
        fn_sens1 = torch.sum(torch.add(predicted[(sensitive!=protected_group)*(predicted.squeeze()!=fav_label)*(labels==fav_label)],1))
        loss = torch.abs(torch.sub(torch.div(tp_sens0,torch.add(tp_sens0,fn_sens0)),torch.div(tp_sens1,torch.add(tp_sens1,fn_sens1))))
    elif metric == 'TNR':
        tn_sens0 = torch.sum(torch.add(predicted[(sensitive==protected_group)*(predicted.squeeze()!=fav_label)*(labels!=fav_label)],1))
        tn_sens1 = torch.sum(torch.add(predicted[(sensitive!=protected_group)*(predicted.squeeze()!=fav_label)*(labels!=fav_label)],1))
        fp_sens0 = torch.sum(predicted[(sensitive==protected_group)*(labels!=fav_label)])
        fp_sens1 = torch.sum(predicted[(sensitive!=protected_group)*(labels!=fav_label)])
        loss = torch.abs(torch.sub(torch.div(tn_sens0,torch.add(tn_sens0,fp_sens0)),torch.div(tn_sens1,torch.add(tn_sens1,fp_sens1))))
    elif metric == 'FPR':
        prob_sens0 = torch.mean(predicted[(sensitive==protected_group)*(labels!=fav_label)].squeeze())
        prob_sens1 = torch.mean(predicted[(sensitive!=protected_group)*(labels!=fav_label)].squeeze())
        loss = torch.abs(torch.sub(prob_sens0, prob_sens1))
    elif metric == 'FNR':
        fn_sens0 = torch.sum(torch.add(predicted[(sensitive==protected_group)*(predicted.squeeze()!=fav_label)*(labels==fav_label)],1))
        fn_sens1 = torch.sum(torch.add(predicted[(sensitive!=protected_group)*(predicted.squeeze()!=fav_label)*(labels==fav_label)],1))
        tp_sens0 = torch.sum(predicted[(sensitive==protected_group)*(labels==fav_label)])
        tp_sens1 = torch.sum(predicted[(sensitive!=protected_group)*(labels==fav_label)])
        loss = torch.abs(torch.sub(torch.div(fn_sens0,torch.add(fn_sens0,tp_sens0)),torch.div(fn_sens1,torch.add(fn_sens1,tp_sens1))))
    elif metric == 'FDR':
        fp_sens0 = torch.sum(predicted[(sensitive==protected_group)*(labels!=fav_label)])
        fp_sens1 = torch.sum(predicted[(sensitive!=protected_group)*(labels!=fav_label)])
        tp_sens0 = torch.sum(predicted[(sensitive==protected_group)*(labels==fav_label)])
        tp_sens1 = torch.sum(predicted[(sensitive!=protected_group)*(labels==fav_label)])
        loss = torch.abs(torch.sub(torch.div(fp_sens0,torch.add(fp_sens0,tp_sens0)),torch.div(fp_sens1,torch.add(fp_sens1,tp_sens1))))
    elif metric == 'FOR':
        fn_sens0 = torch.sum(torch.add(predicted[(sensitive==protected_group)*(predicted.squeeze()!=fav_label)*(labels==fav_label)],1))
        fn_sens1 = torch.sum(torch.add(predicted[(sensitive!=protected_group)*(predicted.squeeze()!=fav_label)*(labels==fav_label)],1))
        tn_sens0 = torch.sum(torch.add(predicted[(sensitive==protected_group)*(predicted.squeeze()!=fav_label)*(labels!=fav_label)],1))
        tn_sens1 = torch.sum(torch.add(predicted[(sensitive!=protected_group)*(predicted.squeeze()!=fav_label)*(labels!=fav_label)],1))
        loss = torch.abs(torch.sub(torch.div(fn_sens0,torch.add(fn_sens0,tn_sens0)),torch.div(fn_sens1,torch.add(fn_sens1,tn_sens1))))
    elif metric == 'SP':
        prob_sens0 = torch.mean(predicted[(sensitive==protected_group)].squeeze()) 
        prob_sens1 = torch.mean(predicted[(sensitive!=protected_group)].squeeze())
        loss = torch.abs(torch.sub(prob_sens0, prob_sens1))
    elif metric == 'EO':
        prob_sens0 = torch.mean(predicted[(sensitive==protected_group)*(labels==fav_label)].squeeze()) 
        prob_sens1 = torch.mean(predicted[(sensitive!=protected_group)*(labels==fav_label)].squeeze())
        loss = torch.abs(torch.sub(prob_sens0, prob_sens1))
    
    params = [ p for p in model.parameters() if p.requires_grad ]
    grad_loss = grad(loss, params, create_graph=True)
    return grad_loss

# Methodology adopted from Wang et al. Counterfactual Distributions...
def getBlackBoxScores(df_min, df, predict_yhat, predict_ytrue, metric, disparity):
    
    p_yhat = predict_yhat(df_min).flatten()
    p_y = predict_ytrue(df_min).flatten()
    p_yhat = np.round(p_yhat.cpu().detach().numpy())
    p_y = p_y.cpu().detach().numpy()
    
    if metric == 'FPR':
        a = np.multiply(p_yhat, 1.0 - p_y)
        b = 1.0 - p_y

        # setup parameters
        var_b = np.square(np.mean(b))
        #if disparity >= 0:
        con_a = np.mean(b) / var_b
        con_b = -np.mean(a) / var_b
        """else:
            con_a = -np.mean(b) / var_b
            con_b = np.mean(a) / var_b"""
        
        # Compute parameters for full data
        p_yhat_all = predict_yhat(df).flatten()
        p_y_all = predict_ytrue(df).flatten()
        p_yhat_all = np.round(p_yhat_all.cpu().detach().numpy())
        p_y_all = p_y_all.cpu().detach().numpy()
        a_all = np.multiply(p_yhat_all, 1.0 - p_y_all)
        b_all = 1.0 - p_y_all
        vals = con_a * a_all + con_b * b_all
    
    elif metric == 'FNR' or metric == 'EO':
        a = np.multiply(1.0 - p_yhat, p_y)
        b = p_y
        
        # setup parameters
        var_b = np.square(np.mean(b))
        #if disparity >= 0:
        con_a = np.mean(b) / var_b
        con_b = -np.mean(a) / var_b 
        """else:
            con_a = -np.mean(b) / var_b
            con_b = np.mean(a) / var_b"""
        
        # Compute parameters for full data
        p_yhat_all = predict_yhat(df).flatten()
        p_y_all = predict_ytrue(df).flatten()
        p_yhat_all = np.round(p_yhat_all.cpu().detach().numpy())
        p_y_all = p_y_all.cpu().detach().numpy().astype(float)
        a_all = np.multiply(1.0 - p_yhat_all, p_y_all)
        b_all = p_y_all
        vals = con_a * a_all + con_b * b_all
    
    elif metric == 'FDR':
        a = np.multiply(p_yhat, 1.0 - p_y)
        b = p_yhat

        # setup parameters
        var_b = np.square(np.mean(b))
        #if disparity >= 0:
        con_a = np.mean(b) / var_b
        con_b = -np.mean(a) / var_b
        """else:
            con_a = -np.mean(b) / var_b
            con_b = np.mean(a) / var_b"""
        
        # Compute parameters for full data
        p_yhat_all = predict_yhat(df).flatten()
        p_y_all = predict_ytrue(df).flatten() 
        p_yhat_all = np.round(p_yhat_all.cpu().detach().numpy())
        p_y_all = p_y_all.cpu().detach().numpy()
        a_all = np.multiply(p_yhat_all, 1.0 - p_y_all)
        b_all = p_y_all
        vals = con_a * a_all + con_b * b_all

    elif metric == 'SP':
        a = np.negative(p_y)
        b = np.mean(p_y)
        
        """if disparity < 0:
            b = np.negative(b)
            a = np.negative(a)"""

        # Compute parameters for full data
        p_y_all = predict_ytrue(df).flatten().cpu().detach().numpy().astype(float)
        a_all = np.negative(p_y_all)
        vals = a_all + b
        
    else:
        return "Invalid Metric"
    
    return vals

def prepX(df, method, outcome_column_name, withInf=True):
    if method == 'test' and withInf: # drop y
        X = df.drop(['influences', outcome_column_name], axis = 1)
    elif method == 'train' and withInf: # keep y
        X = df.drop(['influences'], axis = 1)
    elif method == 'test': # drop y
        X = df.drop([outcome_column_name], axis = 1)
    elif method == 'train': # keep y
        X = df.copy()
        
    return X

def prepY(df, n_method, outcome_column_name):
    if n_method == 'unnormalized':
        outcome = df[outcome_column_name]
        y = df['influences']

    elif n_method == 'scaled':
        outcome = df[outcome_column_name]
        min_inf = df['influences'].min()
        y = (df['influences'] / min_inf)

    else:
        outcome = df[outcome_column_name]
        min_inf = df['influences'].min()
        max_inf = df['influences'].max()
        y = (df['influences'] - min_inf) / (max_inf - min_inf)
        
    return outcome, y

def prepPredictor(model_type, X, y, X_train, y_train, random_state):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=random_state)
    if model_type == 'Kneighbors':
        KNR=KNeighborsRegressor()
        search_grid={'n_neighbors':[x for x in range(20, 100, 20)],'weights':['uniform', 'distance'],'algorithm': 
                     ['auto', 'ball_tree', 'kd_tree', 'brute'], 
                     'metric':['cityblock','cosine', 'euclidean','haversine']}
        search=GridSearchCV(estimator=KNR, param_grid=search_grid, scoring=make_scorer(RBO.prob_to_score),
                            n_jobs=1, cv=cv)
        search.fit(X,y)
        ifs_predictor = KNeighborsRegressor(**search.best_params_)
        ifs_predictor.fit(X_train, y_train)

    elif model_type == 'GradientBoost':
        GBR=GradientBoostingRegressor()
        search_grid={'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1],'max_depth':[1,2,4],
                     'subsample':[.5,.75,1],'random_state':[random_state]}
        search=GridSearchCV(estimator=GBR, param_grid=search_grid, scoring=make_scorer(RBO.prob_to_score),
                            n_jobs=1, cv=cv)
        search.fit(X,y)
        ifs_predictor = GradientBoostingRegressor(**search.best_params_)
        ifs_predictor.fit(X_train, y_train)

    else:
        ifs_predictor = LinearRegression()
        ifs_predictor.fit(X_train, y_train)  
        
    return ifs_predictor

def evalPredictor(model, X_test, y_test):
    pred = model.predict(X_test) 
    rbo_list = []
    iou_list = []
    K_list = []
    minNum = min(101, len(y_test))
    for k in range(1, minNum, 5):
        rbo_list.append(RBO.prob_to_score(y_test.iloc[0:k], pred[0:k]))
        iou_list.append(jaccard_score(np.argsort(y_test).iloc[0:k], np.argsort(pred)[0:k], average='micro'))
        K_list.append(k)
    d = {'K':K_list, 'RBO':rbo_list, 'Jaccard':iou_list}
    perf_tb = pd.DataFrame(d)
    return perf_tb

def calcBlackBoxFn(function_df, apply_df, model, disp_scores, default_minority_group, default_majority_group, subset, 
                 data_name, outcome_column_name, group_column_name, x_apply_tensor = None):
    predict_yhat = lambda x: model(x,1)
    
    # Split dfs up into X, Y, S
    X_function, Y_function, S_function = df_to_XYS(function_df,
                                        outcome_column_name = outcome_column_name,
                                        group_column_name = group_column_name,
                                        minority_group = default_minority_group)
    X_apply, Y_apply, S_apply = df_to_XYS(apply_df,
                                        outcome_column_name = outcome_column_name,
                                        group_column_name = group_column_name,
                                        minority_group = default_minority_group)
    
    # Get rows where sensitive attr is the minority group
    X_function_min = X_function[S_function == 0,:]
    Y_function_min = Y_function[S_function == 0] 
    S_function_min = S_function[S_function == 0]

    function_ds_min = CurrentDataset(X_function_min, Y_function_min)
    functionloader_min = torch.utils.data.DataLoader(function_ds_min, batch_size=100, shuffle=True, num_workers=0)

    # Create true model
    input_dim = functionloader_min.dataset.__getitem__(0)[0].shape[0]
    PATH = data_name +'/'+ data_name +'_model_minority.pth'
    model_min = LogisticRegression(input_dim)
    model_min.cpu()
    train(functionloader_min, model_min)

    predict_ytrue = lambda x: model_min(x)

    # Get the scores based on the metrics
    metrics = ['SP','EO', 'FPR','FDR']
    for metric in metrics:

        disparity = disp_scores.loc[default_minority_group,metric]-disp_scores.loc[default_majority_group,metric]

        values = getBlackBoxScores(torch.from_numpy(X_function_min.astype(int)), torch.from_numpy(X_apply.astype(int)), 
                                   predict_yhat, predict_ytrue, metric, disparity)

        # Save influences to a file
        out = apply_df.copy()
        out['influences'] = values
        out.sort_values(by='influences', key=abs, inplace=True)
        out.to_csv("%s/out/%s_%s_%s_Wang_influences.csv"%(data_name, metric, data_name, subset))

        if x_apply_tensor != None:
            model.eval()
            out['prediction'] = model(x_apply_tensor,1).detach().numpy()
            out.to_csv("%s/out/%s_%s_%s_Wang_influences_withYhat.csv"%(data_name, metric, data_name, subset))
            
def calcBlackBoxMetric(function_df, apply_df, model, disp_scores, default_minority_group, default_majority_group, subset, 
                 data_name, outcome_column_name, group_column_name, metric, x_apply_tensor = None, printOutput = False):
    predict_yhat = lambda x: model(x,1)
    
    # Split dfs up into X, Y, S
    X_function, Y_function, S_function = df_to_XYS(function_df,
                                        outcome_column_name = outcome_column_name,
                                        group_column_name = group_column_name,
                                        minority_group = default_minority_group)
    X_apply, Y_apply, S_apply = df_to_XYS(apply_df,
                                        outcome_column_name = outcome_column_name,
                                        group_column_name = group_column_name,
                                        minority_group = default_minority_group)
    
    # Get rows where sensitive attr is the minority group
    X_function_min = X_function[S_function == 0,:]
    Y_function_min = Y_function[S_function == 0] 
    S_function_min = S_function[S_function == 0]

    function_ds_min = CurrentDataset(X_function_min, Y_function_min)
    functionloader_min = torch.utils.data.DataLoader(function_ds_min, batch_size=100, shuffle=True, num_workers=0)

    # Create true model
    input_dim = functionloader_min.dataset.__getitem__(0)[0].shape[0]
    PATH = data_name +'/'+ data_name +'_model_minority.pth'
    model_min = LogisticRegression(input_dim)
    model_min.cpu()
    train(functionloader_min, model_min)

    predict_ytrue = lambda x: model_min(x)

    # Get the scores based on the metrics
    disparity = disp_scores.loc[default_minority_group,metric]-disp_scores.loc[default_majority_group,metric]

    values = getBlackBoxScores(torch.from_numpy(X_function_min.astype(int)), torch.from_numpy(X_apply.astype(int)), 
                               predict_yhat, predict_ytrue, metric, disparity)

    # Save influences to a file
    if printOutput:
        out = apply_df.copy()
        out['influences'] = values
        out.sort_values(by='influences', key=abs, inplace=True)
        out.to_csv("%s/out/%s_%s_%s_Wang_influences.csv"%(data_name, metric, data_name, subset))

        if x_apply_tensor != None:
            model.eval()
            out['prediction'] = model(x_apply_tensor,1).detach().numpy()
            out.to_csv("%s/out/%s_%s_%s_Wang_influences_withYhat.csv"%(data_name, metric, data_name, subset))
            
    return values

def append_images(images, direction='horizontal',
                  bg_color=(255,255,255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im
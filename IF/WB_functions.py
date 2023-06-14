"""
Functions for computing influence (ground truth and black-box methods)
Author: Brianna Richardson

Script utilized for FAccT '23 paper:


Code built using methods proposed in related works:
Sattigeri, P., Ghosh, S., Padhi, I., Dognin, P., & Varshney, K. R. (2022). Fair Infinitesimal Jackknife: Mitigating the Influence of Biased Training Data Points Without Refitting. Advances in Neural Information Processing Systems. https://doi.org/10.48550/arxiv.2212.06803
Koh, P. W., & Liang, P. (2017). Understanding Black-box Predictions via Influence Functions. Proceedings of the 34th International Conference on Machine Learning, 1885–1894. Retrieved from https://proceedings.mlr.press/v70/koh17a.html
Wang, H., Ustun, B., & Calmon, F. P. (2019). Repairing without Retraining: Avoiding Disparate Impact with Counterfactual Distributions. Proceedings of the 36th International Conference on Machine Learning. Retrieved from http://github.com/ustunb/ctfdist.

These functions heavily influenced by:
- https://github.com/nimarb/pytorch_influence_functions

"""

# Make sure to thank: https://github.com/nimarb/pytorch_influence_functions
# And Koh 2017 paper

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.autograd import grad
from torch.autograd.functional import hessian as hs
#from torch.nn.utils import _stateless
import numpy as np
import pandas as pd

def grad_individual(attr, label, model, loss_fxn):
    """Calculates the gradient z for one sample.
    
    Arguments:
        attr: sample to be given to the Pytorch model
        label: true label for attr
        model: Pytorch model
        loss_fxn: loss function used for prediction/labels
        
    Returns:
        out: call to gradient function which computes gradient of the loss.
    """
    model.eval()
    attr, label = attr.cpu(), torch.unsqueeze(label.float(),0).cpu()
    y = model(attr)
    loss = loss_fxn(y, label)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    return grad(loss, params, create_graph=True)

def grad_group(attr, labels, model, loss_fxn):
    """Calculates the gradient z for several samples.
    
    Arguments:
        attr: data to be given to the Pytorch model
        labels: true labels for attr
        model: Pytorch model
        loss_fxn: loss function used for prediction/labels
        
    Returns:
        grad_loss: gradient of loss with respect to attr
    """
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
    """Calculates the hessian.
    
    Arguments:
        grad_loss: gradient of loss computed from previous functions
        model: Pytorch model
        
    Returns:
        out: hessian computation
    """
    loss_hessian = []
    for i in range(grad_loss.size(0)):
        hs = grad(grad_loss[i], model.parameters(), retain_graph=True)
        hs = torch.vstack([h.view(-1,1) for h in hs])
        loss_hessian.append(hs)
    return torch.hstack(loss_hessian)

def grad_disp(model, metric, attr, labels, sensitive, protected_group=0, fav_label=1):
    """Calculates the gradient with respect to the disparity
    
    Arguments:
        model: Pytorch model
        metric: Metric for which to calculate disparity. See conditionals for options.
        attr: data to be given to the Pytorch model
        labels: true labels for attr
        sensitive: binary (0/1) column with sensitive attribute
        protected_group: int (0 or 1) signifying the identifier for the protected group
                         in sensitive column
        fav_label: int indicating the preferred outcome column value. Should be set to 1. 
        
        
    Returns:
        grad_loss: gradient of loss with respect to disparity of the given metric
    """
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
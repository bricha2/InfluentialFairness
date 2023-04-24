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
from torch.nn.utils import _stateless
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

def calcBlackBoxFn(function_df, apply_df, model, disp_scores, default_minority_group, default_majority_group,
                   outcome_column_name, group_column_name, fn, x_apply_tensor = None):
    """Calculates and outputs black-box scores using Wang et. al's counterfactual distribution method.
        Calculates scores for SP, FNR, FPR, FDR
    
    Arguments:
        function_df: DataFrame containing data to train black box estimator
        apply_df: DataFrame to compute influence for
        model: Pytorch trained model
        disp_scores: DataFrame from calling calc_disp() in support_functions file
        default_minority_group: String value for sensitive labels representing the unprivileged class
        default_majority_group: String value for sensitive labels representing the privileged class
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        fn: file directory to output file with dataframe and influence scores
            _[Metric].csv will be appended to end of fn.
        x_apply_tensor: tensor if you want to add prediction column to output file
        
        
    Returns:
        Nothing
    """
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
    model_min = LogisticRegression(input_dim)
    model_min.cpu()
    train(functionloader_min, model_min)

    predict_ytrue = lambda x: model_min(x)

    # Get the scores based on the metrics
    metrics = ['SP','FNR', 'FPR','FDR']
    for metric in metrics:

        disparity = disp_scores.loc[default_minority_group,metric]-disp_scores.loc[default_majority_group,metric]

        values = getBlackBoxScores(torch.from_numpy(X_function_min.astype(int)), torch.from_numpy(X_apply.astype(int)), 
                                   predict_yhat, predict_ytrue, metric, disparity)
        
        # Save influences to a file
        out = apply_df.copy()
        out['influences'] = values
        out.sort_values(by='influences', key=abs, inplace=True)
        new_fn = "%s_%s.csv"%(fn, metric)
        out.to_csv(new_fn)

        if x_apply_tensor != None:
            model.eval()
            out['prediction'] = model(x_apply_tensor,1).detach().numpy()
            new_fn = "%s_withYhat_%s.csv"%(fn, metric)
            out.to_csv(new_fn)
            
def calcBlackBoxMetric(function_df, apply_df, model, disp_scores, default_minority_group, default_majority_group, 
                       outcome_column_name, group_column_name, metric, x_apply_tensor = None, fn = None):
    """Calculates and outputs black-box scores using Wang et. al's counterfactual distribution method.
        Calculates scores for a given metric
    
    Arguments:
        function_df: DataFrame containing data to train black box estimator
        apply_df: DataFrame to compute influence for
        model: Pytorch trained model
        disp_scores: DataFrame from calling calc_disp() in support_functions file
        default_minority_group: String value for sensitive labels representing the unprivileged class
        default_majority_group: String value for sensitive labels representing the privileged class
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        metric: metric to compute influence for
        fn: file directory to output file with dataframe and influence scores
            _[Metric].csv will be appended to end of fn.
        x_apply_tensor: tensor if you want to add prediction column to output file
        
        
    Returns:
        values: black box influence scores for the apply_df
    """
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
    PATH = data_name +'/'+ data_name +'_model_minority_it'+str(random_seed)+'.pth'
    model_min = LogisticRegression(input_dim)
    model_min.cpu()
    train(functionloader_min, model_min)

    predict_ytrue = lambda x: model_min(x)

    # Get the scores based on the metrics
    disparity = disp_scores.loc[default_minority_group,metric]-disp_scores.loc[default_majority_group,metric]

    values = getBlackBoxScores(torch.from_numpy(X_function_min.astype(int)), torch.from_numpy(X_apply.astype(int)), 
                               predict_yhat, predict_ytrue, metric, disparity)

    # Save influences to a file
    if fn != None:
        out = apply_df.copy()
        out['influences'] = values
        out.sort_values(by='influences', key=abs, inplace=True)
        new_fn = "%s_%s.csv"%(fn, metric)
        out.to_csv(new_fn)

        if x_apply_tensor != None:
            model.eval()
            out['prediction'] = model(x_apply_tensor,1).detach().numpy()
            new_fn = "%s_withYhat_%s.csv"%(fn, metric)
            out.to_csv(new_fn)
            
    return values
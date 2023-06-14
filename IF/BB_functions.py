"""
Functions for computing black-box influence scores
Original Author(s): Hao Wang, Berk Ustun, and Flavio P. Calmon
Modifications to original work by: Brianna Richardson

Script utilized for FAccT '23 paper:


Code adopted from related works:
Wang, H., Ustun, B., & Calmon, F. P. (2019). Repairing without Retraining: Avoiding Disparate Impact with Counterfactual Distributions. Proceedings of the 36th International Conference on Machine Learning. 
Link to GitHub repo: http://github.com/ustunb/ctfdist.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from . import pytorch_functions
import torch

def df_to_XYS(df, outcome_column_name = None, group_column_name = None, minority_group = 0):
    """Splits data into X, Y, S
    
    Arguments:
        df: DataFrame to split
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        minority_group: int value assigned to unprivileged group
        
    Returns:
        Returns df split into X, Y, S
    """
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

def train_test_audit_split(df, train_size, test_size, audit_size, outcome_column_name, group_column_name, random_state):
    """Splits data into train, test, audit subsets
    
    Arguments:
        df: DataFrame to split
        train_size: float fraction of df to go to train
        test_size: float fraction of df to go to test
        audit_size: float fraction of df to go to audit
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        random_state: random_state
        
    Returns:
        Returns df split into train, test, audit
    """

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
    return df_train, df_test, df_audit

def getBlackBoxScores(df_min, df, predict_yhat, predict_ytrue, metric, disparity):
    """Calculates and outputs black-box scores using Wang et. al's counterfactual distribution method.
        Calculates scores for SP, FNR, FPR, FDR
    
    Arguments:
        df_min: DataFrame containing data to train black box estimator
        df: DataFrame to compute influence for
        predict_yhat: function calling original model
        predict_ytrue: function calling model trained on unprivileged group only
        metric: Metric to calculate influence for
        disparity: Disparity of given metric
        
    Returns:
        vals: black box influence scores for given metric
    """
    
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

def calcBlackBoxFn(function_df, apply_df, model, disp_scores, default_minority_group, default_majority_group,
                   subset, data_name, outcome_column_name, group_column_name, x_apply_tensor = None):
    """Calculates and outputs black-box scores using Wang et. al's counterfactual distribution method.
        Calculates scores for SP, FNR, FPR, FDR
    
    Arguments:
        function_df: DataFrame containing data to train black box estimator
        apply_df: DataFrame to compute influence for
        model: Pytorch trained model
        disp_scores: DataFrame from calling calc_disp() in support_functions file
        default_minority_group: String value for sensitive labels representing the unprivileged class
        default_majority_group: String value for sensitive labels representing the privileged class
        subset: String value for subset of the data scores are created for
        data_name: String value for name of the dataset
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
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

    function_ds_min = pytorch_functions.CurrentDataset(X_function_min, Y_function_min)
    functionloader_min = torch.utils.data.DataLoader(function_ds_min, batch_size=100, shuffle=True, num_workers=0)

    # Create true model
    input_dim = functionloader_min.dataset.__getitem__(0)[0].shape[0]
    model_min = pytorch_functions.LogisticRegression(input_dim)
    model_min.cpu()
    pytorch_functions.train(functionloader_min, model_min)

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
        fn = "out/%s_%s_%s_BB_influences%s.csv"%(data_name, metric, subset,"")
        out.to_csv(fn)

        if x_apply_tensor != None:
            model.eval()
            out['prediction'] = model(x_apply_tensor,1).detach().numpy()
            fn = "out/%s_%s_%s_BB_influences%s.csv"%(data_name, metric, subset,"_withYhat")
            out.to_csv(fn)
            
def calcBlackBoxMetric(function_df, apply_df, model, disp_scores, default_minority_group, default_majority_group, 
                       subset, data_name, outcome_column_name, group_column_name, metric, x_apply_tensor = None):
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

    function_ds_min = pytorch_functions.CurrentDataset(X_function_min, Y_function_min)
    functionloader_min = torch.utils.data.DataLoader(function_ds_min, batch_size=100, shuffle=True, num_workers=0)

    # Create true model
    input_dim = functionloader_min.dataset.__getitem__(0)[0].shape[0]
    PATH = 'models/'+ data_name +'_model_minority.pth'
    model_min = pytorch_functions.LogisticRegression(input_dim)
    model_min.cpu()
    pytorch_functions.train(functionloader_min, model_min)

    predict_ytrue = lambda x: model_min(x)

    # Get the scores based on the metrics
    disparity = disp_scores.loc[default_minority_group,metric]-disp_scores.loc[default_majority_group,metric]

    values = getBlackBoxScores(torch.from_numpy(X_function_min.astype(int)), torch.from_numpy(X_apply.astype(int)), 
                               predict_yhat, predict_ytrue, metric, disparity)

    # Save influences to a file
    out = apply_df.copy()
    out['influences'] = values
    out.sort_values(by='influences', key=abs, inplace=True)
    fn = "out/%s_%s_%s_BB_influences%s.csv"%(data_name, metric, subset,"")
    out.to_csv(fn)

    if x_apply_tensor != None:
        model.eval()
        out['prediction'] = model(x_apply_tensor,1).detach().numpy()
        fn = "out/%s_%s_%s_BB_influences%s.csv"%(data_name, metric, subset,"_withYhat")
        out.to_csv(new_fn)
            
    return values
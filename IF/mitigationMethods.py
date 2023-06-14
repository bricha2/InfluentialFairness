"""
Functions for bias mitigation via IF
Author: Brianna Richardson

Script utilized for FAccT '23 paper:



"""
#Read in libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import pickle

from . import pytorch_functions
from . import BB_functions
from . import support_functions

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.autograd import grad
from torch.autograd.functional import hessian as hs
#from torch.nn.utils import _stateless

import sys
import configparser

def retrainModel(temp, metric, outcome_column_name, group_column_name, default_minority_group,
                default_majority_group, x_eval_tensor, y_eval_tensor, s_eval_tensor):
    """Retrains the model with the new DataFrame
    
    Arguments:
        temp: DataFrame new training data
        metric: to calculate influence for
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        default_minority_group: String value for sensitive labels representing the unprivileged class
        default_majority_group: String value for sensitive labels representing the privileged class
        x_eval_tensor: tensor for data to test the new model
        y_eval_tensor: tensor for labels to test the new model
        s_eval_tensor: tensor for sensitive labels to test the new model
        
        
    Returns:
        y_model: Logistic Regression new trained LR model
        disparity: int new disparity of given metric
        accuracy: int new accuracy of new model
        
    """
    # Re-prep train for model
    X_temp, Y_temp, S_temp = BB_functions.df_to_XYS(temp,
                                          outcome_column_name = outcome_column_name,
                                          group_column_name = group_column_name,
                                          minority_group = default_minority_group)

    temp_ds = pytorch_functions.CurrentDataset(X_temp, Y_temp)

    temploader = torch.utils.data.DataLoader(temp_ds, batch_size=100,
                                                              shuffle=True, num_workers=0)

    # Create model
    input_dim = temploader.dataset.__getitem__(0)[0].shape[0]
    y_model = pytorch_functions.LogisticRegression(input_dim)
    y_model.cpu()
    pytorch_functions.train(temploader, y_model)

    # Calculate disparities
    disp_scores = support_functions.calc_disp(y_model, x_eval_tensor, y_eval_tensor, s_eval_tensor, 
                                            [default_minority_group,default_majority_group])
    disparity = disp_scores.loc[default_majority_group,metric]- disp_scores.loc[default_minority_group,metric]
    acccuracy = disp_scores.loc['All','ACC'].flat[0]
    
    return y_model, disparity, accuracy

def removal(inf_df, topn, metric, outcome_column_name, group_column_name, default_minority_group,
           default_majority_group, x_eval_tensor, y_eval_tensor, s_eval_tensor):
    """Removes top n opponents.
    
    Arguments:
        inf_df: DataFrame original data
        topn: int or String containing number for top points to remove or 'knee' indicating to use the knee approach
        metric: to calculate influence for
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        default_minority_group: String value for sensitive labels representing the unprivileged class
        default_majority_group: String value for sensitive labels representing the privileged class
        x_eval_tensor: tensor for data to test the new model
        y_eval_tensor: tensor for labels to test the new model
        s_eval_tensor: tensor for sensitive labels to test the new model
        
        
    Returns:
        y_model: Logistic Regression new trained LR model
        temp: DataFrame new df model
        disparity: int new disparity of given metric
        accuracy: int new accuracy of new model
        mean_inf: int mean influence of points removed
        sum_inf:  int sum of influence of points removed
        topn: int number of points removed
        
    """
    
    # Be sure to rank dataframe by influence
    inf_df = inf_df.sort_values(by ='influences')
    
    # Extract opponents
    opp_df = inf_df[inf_df['influences'] > 0]
    
    # Identify method to find topn
    if topn == 'knee':
        # Do the knee approach
        expandingMean = opp_df['influences'].iloc[::-1].expanding().mean()
        topn = support_functions.getKnee(expandingMean, "convex", "decreasing", S=75)
        if topn == None:
            topn = 20 if len(opp_df) > 20 else len(opp_df)
        else:
            # Just start with the top 20
            topn = 20 if len(opp_df) > 20 else len(opp_df)
            
    
    # Extract information about top n samples
    mean_inf = inf_df['influences'].tail(topn).mean()
    sum_inf = sum(inf_df['influences'].tail(topn))
    
    # Now make the modifications to the training data
    topm = len(inf_df) - topn
    temp = inf_df.head(topm).drop(['influences'], axis=1)
       
    y_model, disparity, accuracy =  retrainModel(temp, metric, outcome_column_name, group_column_name, 
                                                 default_minority_group,default_majority_group, 
                                                 x_eval_tensor, y_eval_tensor, s_eval_tensor)

    return y_model, temp, disparity, accuracy, mean_inf, sum_inf, topn
                
    
def duplication(inf_df, topn, metric, outcome_column_name, group_column_name, default_minority_group,
           default_majority_group, x_eval_tensor, y_eval_tensor, s_eval_tensor):
    """duplicates top n proponents.
    
    Arguments:
        inf_df: DataFrame original data
        topn: int or String containing number for top points to duplicate or 'knee' indicating to use the knee approach
        metric: to calculate influence for
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        default_minority_group: String value for sensitive labels representing the unprivileged class
        default_majority_group: String value for sensitive labels representing the privileged class
        x_eval_tensor: tensor for data to test the new model
        y_eval_tensor: tensor for labels to test the new model
        s_eval_tensor: tensor for sensitive labels to test the new model
        
        
    Returns:
        y_model: Logistic Regression new trained LR model
        temp: DataFrame new df model
        disparity: int new disparity of given metric
        accuracy: int new accuracy of new model
        mean_inf: int mean influence of points removed
        sum_inf:  int sum of influence of points removed
        topn: int number of points removed
        
    """
    
    # Be sure to rank dataframe by influence
    inf_df = inf_df.sort_values(by ='influences')
    df_train = inf_df.drop(['influences'], axis=1)
    
    # Extract proponents
    prop_df = inf_df[inf_df['influences'] < 0]
    
    # Identify method to find topn
    if topn == 'knee':
        # Do the knee approach
        expandingMean = short_df['influences'].expanding().mean()
        topn = support_functions.getKnee(expandingMean, "concave", "increasing", S=75)
        if topn == None:
            topn = 20 if len(prop_df) > 20 else len(prop_df)
        else:
            # Just start with the top 20
            topn = 20 if len(prop_df) > 20 else len(prop_df)
            
    
    # Extract information about top n samples
    mean_inf = inf_df['influences'].head(topn).mean()
    sum_inf = sum(inf_df['influences'].head(topn))
    
    # Now make the modifications to the training data
    temp = inf_df.head(topn).drop(['influences'], axis=1)
    temp = pd.concat([df_train, temp])
       
    y_model, disparity, accuracy =  retrainModel(temp, metric, outcome_column_name, group_column_name, 
                                                 default_minority_group,default_majority_group, 
                                                 x_eval_tensor, y_eval_tensor, s_eval_tensor)

    return y_model, temp, disparity, accuracy, mean_inf, sum_inf, topn

def invNearestOpp(inf_df, topn, metric, outcome_column_name, group_column_name, default_minority_group,
           default_majority_group, x_eval_tensor, y_eval_tensor, s_eval_tensor):
    """duplicates the proponents most similar to the top n opponents
    
    Arguments:
        inf_df: DataFrame original data
        topn: int or String containing number for top points to duplicate or 'knee' indicating to use the knee approach
        metric: to calculate influence for
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        default_minority_group: String value for sensitive labels representing the unprivileged class
        default_majority_group: String value for sensitive labels representing the privileged class
        x_eval_tensor: tensor for data to test the new model
        y_eval_tensor: tensor for labels to test the new model
        s_eval_tensor: tensor for sensitive labels to test the new model
        
        
    Returns:
        y_model: Logistic Regression new trained LR model
        temp: DataFrame new df model
        disparity: int new disparity of given metric
        accuracy: int new accuracy of new model
        mean_inf: int mean influence of points removed
        sum_inf:  int sum of influence of points removed
        topn: int number of points removed
        
    """
    # Be sure to rank dataframe by influence
    inf_df = inf_df.sort_values(by ='influences')
    df_train = inf_df.drop(['influences'], axis=1)
    
    # Extract opponents
    opp_df = inf_df[inf_df['influences'] > 0]
    
    # Identify method to find topn
    if topn == 'knee':
        # Do the knee approach
        expandingMean = opp_df['influences'].iloc[::-1].expanding().mean()
        topn = support_functions.getKnee(expandingMean, "convex", "decreasing", S=75)
        if topn == None:
            topn = 20 if len(opp_df) > 20 else len(opp_df)
        else:
            # Just start with the top 20
            topn = 20 if len(opp_df) > 20 else len(opp_df)
            
    # Extract information about top n samples
    mean_inf = inf_df['influences'].tail(topn).mean()
    sum_inf = sum(temp_df['influences'].tail(topn))
            
    temp = inf_df.tail(topn).drop(['influences'], axis=1)

    # Negate the outcome column and then extend it
    df_train_extended = df_train.copy()
    df_train_extended[outcome_column_name] = [0 if x == 1 else 1 for x in df_train_extended[outcome_column_name]]
    n = df_train_extended.shape[1]
    df_train_extended = df_train_extended.assign(**{f'C{k}': df_train_extended[outcome_column_name] 
                                                    for k in range(1, n+1)})
    temp_extended = temp.assign(**{f'C{j}': temp[outcome_column_name] for j in range(1, n+1)})

    # Build a KDTree
    tree = KDTree(df_train_extended, leaf_size=df_train_extended.shape[0]+1)
    query = lambda x: tree.query(x.values.reshape(1, -1), k=1)[1][0]

    # Get the rows from df_train with the closest fit
    temp = df_train.iloc[temp_extended.apply(query, axis = 1)]

    # Add them to df_train
    temp = pd.concat([df_train, temp])

    y_model, disparity, accuracy =  retrainModel(temp, metric, outcome_column_name, group_column_name, 
                                                 default_minority_group,default_majority_group, 
                                                 x_eval_tensor, y_eval_tensor, s_eval_tensor)

    return y_model, temp, disparity, accuracy, mean_inf, sum_inf, topn
                
    
def relabelOpp(inf_df, topn, metric, outcome_column_name, group_column_name, default_minority_group,
           default_majority_group, x_eval_tensor, y_eval_tensor, s_eval_tensor):
    """Relabels top n opponents.
    
    Arguments:
        inf_df: DataFrame original data
        topn: int or String containing number for top points to remove or 'knee' indicating to use the knee approach
        metric: to calculate influence for
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        default_minority_group: String value for sensitive labels representing the unprivileged class
        default_majority_group: String value for sensitive labels representing the privileged class
        x_eval_tensor: tensor for data to test the new model
        y_eval_tensor: tensor for labels to test the new model
        s_eval_tensor: tensor for sensitive labels to test the new model
        
        
    Returns:
        y_model: Logistic Regression new trained LR model
        temp: DataFrame new df model
        disparity: int new disparity of given metric
        accuracy: int new accuracy of new model
        mean_inf: int mean influence of points removed
        sum_inf:  int sum of influence of points removed
        topn: int number of points removed
        
    """
    
    # Be sure to rank dataframe by influence
    inf_df = inf_df.sort_values(by ='influences')
    df_train = inf_df.drop(['influences'], axis=1)
    
    # Extract opponents
    opp_df = inf_df[inf_df['influences'] > 0]
    
    # Identify method to find topn
    if topn == 'knee':
        # Do the knee approach
        expandingMean = opp_df['influences'].iloc[::-1].expanding().mean()
        topn = support_functions.getKnee(expandingMean, "convex", "decreasing", S=75)
        if topn == None:
            topn = 20 if len(opp_df) > 20 else len(opp_df)
        else:
            # Just start with the top 20
            topn = 20 if len(opp_df) > 20 else len(opp_df)
            
    
    # Extract information about top n samples
    mean_inf = inf_df['influences'].tail(topn).mean()
    sum_inf = sum(inf_df['influences'].tail(topn))
    
    # Now make the modifications to the training data
    topn_samples = inf_df.tail(topn).drop(['influences'], axis=1)
    opponents = df_train[df_train.index.isin(topn_samples.index)]
    temp_train = df_train[~df_train.index.isin(topn_samples.index)]
    ind_set = np.append(ind_set,topn_samples.index)
    ind_set = np.unique(ind_set)
    
    # Reverse the opponents and add
    opponents[outcome_column_name] = [1 if x == 0 else 0 for x in opponents[outcome_column_name]]
    temp = pd.concat([temp_train, opponents])
       
    y_model, disparity, accuracy =  retrainModel(temp, metric, outcome_column_name, group_column_name, 
                                                 default_minority_group,default_majority_group, 
                                                 x_eval_tensor, y_eval_tensor, s_eval_tensor)

    return y_model, temp, disparity, accuracy, mean_inf, sum_inf, topn

def recommend(inf_df, ext_df, r_method, topn, metric, outcome_column_name, group_column_name, default_minority_group,
           default_majority_group, x_eval_tensor, y_eval_tensor, s_eval_tensor):
    """Trains a rule regressor using influence scores and then add points from external df based on rules.
    
    Arguments:
        inf_df: DataFrame original data
        ext_df: DataFrame external data that will be added to original data
        r_method: String 'LRR' or 'BRCG' depicting rule method to use
        topn: int or String containing number for top points to remove or 'knee' indicating to use the knee approach
        metric: to calculate influence for
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        default_minority_group: String value for sensitive labels representing the unprivileged class
        default_majority_group: String value for sensitive labels representing the privileged class
        x_eval_tensor: tensor for data to test the new model
        y_eval_tensor: tensor for labels to test the new model
        s_eval_tensor: tensor for sensitive labels to test the new model
        
        
    Returns:
        y_model: Logistic Regression new trained LR model
        temp: DataFrame new df model
        disparity: int new disparity of given metric
        accuracy: int new accuracy of new model
        mean_inf: int mean influence of points removed
        sum_inf:  int sum of influence of points removed
        topn: int number of points removed
        
    """
    
    # Be sure to rank dataframe by influence
    inf_df = inf_df.sort_values(by ='influences')
    
    # Identify method to find topn
    if topn == 'knee':
        # Do the knee approach
        expandingMean = inf_df['influences'].expanding().mean()
        topn = support_functions.getKnee(expandingMean, "concave", "increasing", S=75)
        if topn == None:
            topn = 20 if len(inf_df) > 20 else len(inf_df)
        else:
            # Just start with the top 20
            topn = 20 if len(inf_df) > 20 else len(inf_df)
            
    topn_val = inf_df.iloc[topn]['influences']
    inf_df['bin_inf'] = [0 if x > topn_val else 1 for x in inf_df['influences']]
    
    # split into train/test
    df_train_inf, df_test_inf = train_test_split(inf_df,
                                         train_size = 0.6,
                                         stratify = df_inf[[group_column_name, outcome_column_name]])

    # Split into X and y
    X_train = df_train_inf.drop(['influences', 'bin_inf'],axis=1)
    Y_train = df_train_inf['bin_inf']
    Y_train_raw = df_train_inf['influences']
    X_test = df_test_inf.drop(['influences', 'bin_inf'],axis=1)
    Y_test = df_test_inf['bin_inf']
    Y_test_raw = df_test_inf['influences']

    # Apply to ifs predictor
    gb_preds_b = predict_bin(ifs_predictor,X_test)

    # Binarize Features
    fb = FeatureBinarizer(negations=True)
    X_train_b = fb.fit_transform(X_train)
    X_test_b = fb.transform(X_test)
    
    if r_method == 'BRCG':

        # BRCG
        brcg_explain = fit_predict_bcrg(X_train_b, Y_train, X_test_b, Y_test)
        
        # Get rules
        rules = brcg_explain['rules']
            
    elif r_method == 'LRR':

        # LOGRR
        logrr_explain = fit_predict_logrr(X_train_b, Y_train, X_test_b, Y_test)
        expandingMean = abs(logrr_explain['coefficient'][1:]).expanding().mean()
        topn = support_functions.getKnee(expandingMean, "convex", "decreasing", S=2)

        # Get rules
        rules = logrr_explain[1:topn+1]['rule'][logrr_explain[1:topn+1]['coefficient']>0]

    matching_samples_test = getMatchingSamples(rules, ext_df)
    n_points = len(matching_samples_test)
    temp = pd.concat([df_train, matching_samples_test])
    
    y_model, disparity, accuracy =  retrainModel(temp, metric, outcome_column_name, group_column_name, 
                                                 default_minority_group,default_majority_group, 
                                                 x_eval_tensor, y_eval_tensor, s_eval_tensor)

    return y_model, temp, disparity, accuracy, mean_inf, sum_inf, n_points

def addition(inf_df, ext_df, topn, metric, outcome_column_name, group_column_name, default_minority_group,
           default_majority_group, x_eval_tensor, y_eval_tensor, s_eval_tensor):
    """Adds points from external data using influence estimator
    
    Arguments:
        inf_df: DataFrame original data
        ext_df: DataFrame external data that already includes influence estimations
        topn: int or String containing number for top points to remove or 'knee' indicating to use the knee approach
        metric: to calculate influence for
        outcome_column_name: String column name for true labels
        group_column_name: String column name for sensitive labels
        default_minority_group: String value for sensitive labels representing the unprivileged class
        default_majority_group: String value for sensitive labels representing the privileged class
        x_eval_tensor: tensor for data to test the new model
        y_eval_tensor: tensor for labels to test the new model
        s_eval_tensor: tensor for sensitive labels to test the new model
        
        
    Returns:
        y_model: Logistic Regression new trained LR model
        temp: DataFrame new df model
        disparity: int new disparity of given metric
        accuracy: int new accuracy of new model
        mean_inf: int mean influence of points removed
        sum_inf:  int sum of influence of points removed
        topn: int number of points removed
        
    """
    
    # Be sure to rank dataframe by influence
    ext_df = ext_df.sort_values(by ='influences')
    
    # Extract proponents
    prop_df = ext_df[ext_df['influences'] < 0]
    
    # Identify method to find topn
    if topn == 'knee':
        # Do the knee approach
        expandingMean = short_df['influences'].expanding().mean()
        topn = support_functions.getKnee(expandingMean, "concave", "increasing", S=75)
        if topn == None:
            topn = 20 if len(prop_df) > 20 else len(prop_df)
        else:
            # Just start with the top 20
            topn = 20 if len(prop_df) > 20 else len(prop_df)
            
    mean_inf = ext_df['influences'].head(topn).mean()
    sum_inf = sum(ext_df['influences'].head(topn))
    temp = ext_df.head(topn).drop(['influences'], axis=1)

    # Add those top i points to df_train
    temp = pd.concat([inf_df.drop(['influences'], axis=1), temp])
    
    y_model, disparity, accuracy =  retrainModel(temp, metric, outcome_column_name, group_column_name, 
                                                 default_minority_group,default_majority_group, 
                                                 x_eval_tensor, y_eval_tensor, s_eval_tensor)

    return y_model, temp, disparity, accuracy, mean_inf, sum_inf, topn
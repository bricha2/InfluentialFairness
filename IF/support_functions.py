"""
Support functions for Influential Fairness analysis
Author: Brianna Richardson

Script utilized for FAccT '23 paper:


"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score,make_scorer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split, GridSearchCV
import torch

from pathlib import Path
from os.path import exists
#from PIL import Image
from kneed import KneeLocator

from . import RBO_functions

def calc_disp(model, attr, labels, sensitive, sensitive_groups = ['0','1']):
    """Generates a table with scores for each sensitive group and across all samples.
    Metrics calculated include: TPR, TNR, FNR, FPR, FDR, FOR, EO, SP, ACC.
    
    Arguments:
        model: pytorch model
        attr: data to calculate metrics on 
        labels: true labels for attr
        sensitive: binarized (0/1) sensitive group assignment for attr
        sensitive_groups: list with two elements indicating dataset specific terminology for minority and majority, respectively
        
    Returns:
        scores: DataFrame with scores for each sensitive group and all samples"""
        
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

def calc_disp_nomodel(labels, preds, sensitive):
    """Generates a table with scores for each sensitive group and across all samples.
    Metrics calculated include: TPR, TNR, FNR, FPR, FDR, FOR, EO, SP, ACC.
    Does not compute disparity from model (like previous function).
    
    Arguments:
        labels: list true labels
        preds: list of (binary) predictions
        sensitive: binarized (0/1) sensitive group assignment
        
    Returns:
        scores: DataFrame with scores for each sensitive group and all samples"""
    
    sensitive_groups = ['0','1']

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

def prepX(df, method, outcome_column_name, withInf=True):
    """Prep data to be given to model.
    
    Arguments:
        df: DataFrame to extract relevant data to be fed into predictor 
        method: String that is either 'train' or 'test'. 'train' method keeps
                outcome column, while 'test' method removes outcome column
        outcome_column_name: String with name of column that holds true value
        withInf: Boolean that depicts if influence score is in the df
        
    Returns:
        X: DataFrame with attributes to be fed into model"""
    
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
    """Prep influence according to normalization method
    
    Arguments:
        df: DataFrame to extract relevant data for predictor 
        n_method: String that is 'unnormalized', 'scaled' or other. 
                  'unnormalized' method keeps influence column the same.
                  'scaled' method scales influence column by min val
                  any other string will lead to a min-max scaling
        outcome_column_name: String with name of column that holds true value
        withInf: Boolean that depicts if influence score is in the df
        
    Returns:
        outcome: outcome column
        y: scaled influence column"""
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
    """Train predictor using CV and grid search.
    
    Arguments:
        model_type: String that is 'Kneighbors', 'GradientBoost' or other. 
                  'Kneighbors' trains a Kneighbors regressor.
                  'GradientBoost' trains a Gradient Boosting regressor
                  any other string will train a Linear Ridge Regressor
        X,y: data, labels used for grid search
        X_train, y_train: data, labels used to train model 
        random_state: random_state for predictor
        
    Returns:
        ifs_predictor: Trained predictor for predicting influence"""
    
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=random_state)
    if model_type == 'Kneighbors':
        KNR=KNeighborsRegressor()
        search_grid={'n_neighbors': [x for x in range(20, 100, 20)],
                     'weights': ['uniform', 'distance'],
                     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
                     'metric': ['cityblock','cosine', 'euclidean','haversine']}
        search=GridSearchCV(estimator=KNR, param_grid=search_grid, scoring=make_scorer(RBO_functions.bal_rbo),
                            n_jobs=1, cv=cv)
        search.fit(X,y)
        ifs_predictor = KNeighborsRegressor(**search.best_params_)
        ifs_predictor.fit(X_train, y_train)

    elif model_type == 'GradientBoost':
        GBR=GradientBoostingRegressor()
        search_grid={'n_estimators':[500,1000,2000],
                     'learning_rate':[.001,0.01,.1],
                     'max_depth':[1,2,4],
                     'subsample':[.5,.75,1],
                     'random_state':[random_state]}
        search=GridSearchCV(estimator=GBR, param_grid=search_grid, scoring=make_scorer(RBO_functions.bal_rbo),
                            n_jobs=1, cv=cv)
        search.fit(X,y)
        ifs_predictor = GradientBoostingRegressor(**search.best_params_)
        ifs_predictor.fit(X_train, y_train)

    else:
        RR = Ridge()
        search_grid = {
            'solver':['svd', 'cholesky', 'lsqr', 'sag'],
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
            'fit_intercept':[True, False],
        }
        search=GridSearchCV(estimator=RR, param_grid=search_grid, scoring=make_scorer(RBO_functions.bal_rbo),
                            n_jobs=1, cv=cv)
        search.fit(X,y)
        ifs_predictor = Ridge(**search.best_params_)
        ifs_predictor.fit(X_train, y_train)
        
    return ifs_predictor

def evalPredictor(model, X_test, y_test):
    """Evaluate a trained predictor with a given test set, using balanced RBO and Jaccard (IOU).
       Trains the test distribution in increments of 5.
    
    Arguments:
        model: Trained influence regressor
        X_test: Test distribution to be given to regressor
        y_test: True labels for test distribution
        
    Returns:
        perf_tb: DataFrame with scores calculated in increments of 5"""
    pred = model.predict(X_test) 
    rbo_list = []
    iou_list = []
    K_list = []
    minNum = min(101, len(y_test))
    for k in range(1, minNum, 5):
        rbo_list.append(RBO_functions.bal_rbo(y_test.iloc[0:k], pred[0:k]))
        iou_list.append(jaccard_score(np.argsort(y_test).iloc[0:k], np.argsort(pred)[0:k], average='micro'))
        K_list.append(k)
    d = {'K':K_list, 'RBO':rbo_list, 'Jaccard':iou_list}
    perf_tb = pd.DataFrame(d)
    return perf_tb

def getKnee(expandingMean, curve, direction, S=75):
    """Calculates the knee of the curve using the KneeLocator
    
    Arguments:
        expandingMean: list containing expanding means of a list of numbers
        curve: String 'concave' vs 'convex'. See KneeLocator documentation for more details
        direction: String 'increasing' vs 'decreasing'. See KneeLocator documentation for more details
        
    Returns:
        topn: int containing the number of items within the knee of the curve"""
    if curve == 'concave':
        alt_curve = 'convex'
    else:
        alt_curve = 'concave'
    
    kl_prim = KneeLocator([*range(1,len(expandingMean)+1)],expandingMean, curve=curve, S=S,
                                                  direction = direction)
    kl_prim_knee = kl_prim.knee
    kl_sec = KneeLocator([*range(1,len(expandingMean)+1)],expandingMean, curve=alt_curve, S=S,
                                                  direction = direction)
    kl_sec_knee = kl_sec.knee

    if kl_prim_knee == 1:
        if kl_sec_knee != None and kl_sec_knee > 1:
            topn = kl_sec_knee
        elif len(expandingMean) > 0:
            print("%s was 1 and %s was less, so knee will be 20 or less."%(curve, alt_curve))
            print(expandingMean)
            topn = 20 if len(expandingMean) >= 20 else len(expandingMean)
        else:
            print("%s was 1 and %s were None and expanding mean was empty, so breaking."%(curve, alt_curve))
            return None
    elif kl_prim_knee == None:
        if kl_sec_knee != None:
            print("%s was None and %s was not."%(curve, alt_curve))
            print(expandingMean)
            topn = kl_sec_knee
        elif len(expandingMean) > 0:
            print("Both %s and %s were None, just grabbing the top 20 or less."%(curve, alt_curve))
            print(expandingMean)
            topn = 20 if len(expandingMean) >= 20 else len(expandingMean)
        else:
            print("Both %s and %s were None and expanding mean was empty, so breaking."%(curve, alt_curve))
            return None
    else:
        topn = kl_prim_knee
    return topn
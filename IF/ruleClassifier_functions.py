"""
Functions for building and evaluating rule based classifiers
Original Author(s): Floid Gilbert
Modifications to original work by: Brianna Richardson

Script utilized for FAccT '23 paper:

Implementing code from Github 
Link to GitHub repo: https://github.com/Trusted-AI/AIX360/blob/master/examples/rbm/feature_binarizer_from_trees.ipynb

"""
import sys
sys.path.insert(0, 'AIX360')

from aix360.algorithms.rbm import BRCGExplainer, BooleanRuleCG, FeatureBinarizer, LogisticRuleRegression, GLRMExplainer, LinearRuleRegression
from sklearn.model_selection import train_test_split
import configparser
import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import pickle

random_seed = 5968
random_state = np.random.RandomState(random_seed)

def print_brcg_rules(rules):
    print('Predict Y=1 if ANY of the following rules are satisfied, otherwise Y=0:\n')
    for rule in rules:
        print(f'  - {rule}')
    print()

def fit_predict_bcrg(X_train_b, y_train, X_test_b, y_test, lambda0=0.001, lambda1=0.001):
    bcrg = BooleanRuleCG(lambda0, lambda1, silent=True)
    explainer = BRCGExplainer(bcrg)
    t = time()
    explainer.fit(X_train_b, y_train)
    print(f'Model trained in {time() - t:0.1f} seconds\n')
    print_metrics(y_test, explainer.predict(X_test_b))
    print_brcg_rules(explainer.explain()['rules'])
    return explainer.explain()
    
def fit_predict_logrr(X_train_b, y_train, X_test_b, y_test):
    logrr = LogisticRuleRegression(lambda0=0.005, lambda1=0.001, maxSolverIter=1000)
    explainer = GLRMExplainer(logrr)
    t = time()
    explainer.fit(X_train_b, y_train)
    print(f'Model trained in {time() - t:0.1f} seconds\n')
    print_metrics(y_test, explainer.predict(X_test_b))
    return explainer.explain()

def fit_predict_linrr(X_train_b, y_train, X_test_b, y_test):
    linrr = LinearRuleRegression(lambda0=0.005, lambda1=0.001)
    explainer = GLRMExplainer(linrr)
    t = time()
    explainer.fit(X_train_b, y_train)
    print(f'Model trained in {time() - t:0.1f} seconds\n')
    print_metrics_cont(y_test, explainer.predict(X_test_b))
    return explainer.explain()

def print_metrics(y_truth, y_pred):
    print(f'Accuracy = {accuracy_score(y_truth, y_pred):0.5f}')
    print(f'Precision = {precision_score(y_truth, y_pred):0.5f}')
    print(f'Recall = {recall_score(y_truth, y_pred):0.5f}')
    print(f'F1 = {f1_score(y_truth, y_pred):0.5f}')
    print(f'F1 Weighted = {f1_score(y_truth, y_pred, average="weighted"):0.5f}')
    print(f'Balanced Accuracy = {balanced_accuracy_score(y_truth, y_pred):0.5f}')
    print()
    
def print_metrics_cont(y_truth, y_pred):
    print(f'MSE = {mean_squared_error(y_truth, y_pred):0.5f}')
    print(f'MAE = {mean_absolute_error(y_truth, y_pred):0.5f}')
    print(f'R2 = {r2_score(y_truth, y_pred):0.5f}')
    print()
    
def getMatchingSamples(exp_rules, df):
    matching_samples = pd.DataFrame()
    # Parse the rules and find the samples that match
    for rule in exp_rules:
        # split by AND
        rule = rule.split("AND")
        all_samples = df.copy()

        # For each item in rule
        for r in rule:

            # split by space
            comps = r.split('not')

            # if length greater than one
            if len(comps) > 1:

                # Get samples where columns is 0
                all_samples = all_samples.loc[all_samples[comps[0].strip()] == 0]

            else:
                
                # Split by greater than or less than
                comps_less = r.split('<')
                comps_greater = r.split('>')
                comps_lesseq = r.split('<=')
                comps_greatereq = r.split('>=')
                
                # if length greater than one
                if len(comps_lesseq) > 1:

                    # Get samples where columns is 0
                    all_samples = all_samples.loc[all_samples[comps_lesseq[0].strip()] <= float(comps_lesseq[1].strip())]
                
                # if length greater than one
                elif len(comps_greatereq) > 1:

                    # Get samples where columns is 0
                    all_samples = all_samples.loc[all_samples[comps_greatereq[0].strip()] >= float(comps_greatereq[1].strip())]
                
                elif len(comps_less) > 1:

                    # Get samples where columns is 0
                    all_samples = all_samples.loc[all_samples[comps_less[0].strip()] < float(comps_less[1].strip())]
                
                # if length greater than one
                elif len(comps_greater) > 1:

                    # Get samples where columns is 0
                    all_samples = all_samples.loc[all_samples[comps_greater[0].strip()] > float(comps_greater[1].strip())]
                    
                else:
                
                    # Get samples where columns is 1
                    all_samples = all_samples.loc[all_samples[r.strip()] == 1]

        # Add as matchinga samples
        matching_samples = pd.concat([matching_samples, all_samples], axis=0)

    # remove duplicates
    matching_samples = matching_samples.reset_index().drop_duplicates(subset='index').set_index('index')
    
    return matching_samples

def predict_bin(ifs_predictor, inputs):

    gb_preds = ifs_predictor.predict(inputs)
    gb_preds_b = gb_preds.copy()
    gt_ind = gb_preds_b > topn_val
    lt_ind = gb_preds_b < topn_val
    gb_preds_b[gt_ind] = 0
    gb_preds_b[lt_ind] = 1
    
    return gb_preds_b
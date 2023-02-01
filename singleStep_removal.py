# Read in libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import pickle

from influence_functions import *

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
from torch.nn.utils import _stateless

random_seed = 5968
random_state = np.random.RandomState(random_seed)

import sys
import configparser

keep_old = False

if __name__ == "__main__":
    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    data_name = config.get('SETTINGS','data_name')
    outcome_column_name = config.get('SETTINGS','outcome_column_name')
    group_column_name = config.get('SETTINGS','group_column_name')
    default_minority_group = config.get('SETTINGS','default_minority_group')
    default_majority_group = config.get('SETTINGS','default_majority_group')
            
    metrics = ['SP','EO', 'FPR', 'FDR']

    # Get metric disparities
    fn = "%s/out/%s_eval_disparities.csv"%(data_name,data_name)
    st_df = pd.read_csv(fn, sep = ',', index_col=0)
    st_df = st_df.iloc[: , 4:]
    disp = st_df.loc[default_majority_group] - st_df.loc[default_minority_group]
    del st_df

    # Read in the original train
    df_train = pd.read_csv("%s/out/%s_train.csv"%(data_name,data_name), index_col = 0)

    # prep the original train data
    X_train, Y_train, S_train = df_to_XYS(df_train, outcome_column_name = outcome_column_name,
                                       group_column_name = group_column_name,
                                       minority_group = default_minority_group)
    train_ds = CurrentDataset(X_train, Y_train)
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=100, shuffle=True, num_workers=0)
    input_dim = trainloader.dataset.__getitem__(0)[0].shape[0]
    df_train[group_column_name] = [0 if x == default_minority_group else 1 for x in df_train[group_column_name]]

    # Read in the original model
    model = load_model(input_dim, '%s/%s_model.pth'%(data_name,data_name))

    # Make train tensors
    y_train_tensor = torch.tensor(Y_train.astype(np.float64),requires_grad=True).long()
    x_train_tensor = torch.tensor(X_train.astype(np.float32),requires_grad=True)
    s_train_tensor = torch.tensor(S_train.astype(np.float32),requires_grad=True)
    
    # Read in the public
    df_public = pd.read_csv("%s/out/%s_public.csv"%(data_name,data_name), index_col = 0)

    # Prep the public & make tensors
    X_public, Y_public, S_public = df_to_XYS(df_public, outcome_column_name = outcome_column_name,
                                       group_column_name = group_column_name,
                                       minority_group = default_minority_group)
    public_ds = CurrentDataset(X_public, Y_public)
    publicloader = torch.utils.data.DataLoader(public_ds, batch_size=100, shuffle=True, num_workers=0)
    y_public_tensor = torch.tensor(Y_public.astype(np.float64),requires_grad=True).long()
    x_public_tensor = torch.tensor(X_public.astype(np.float32),requires_grad=True)
    s_public_tensor = torch.tensor(S_public.astype(np.float32),requires_grad=True)

    # Read in the eval
    df_eval = pd.read_csv("%s/out/%s_eval.csv"%(data_name,data_name), index_col = 0)

    # Prep the eval & make tensors
    X_eval, Y_eval, S_eval = df_to_XYS(df_eval, outcome_column_name = outcome_column_name,
                                       group_column_name = group_column_name,
                                       minority_group = default_minority_group)
    eval_ds = CurrentDataset(X_eval, Y_eval)
    evalloader = torch.utils.data.DataLoader(eval_ds, batch_size=100, shuffle=True, num_workers=0)
    y_eval_tensor = torch.tensor(Y_eval.astype(np.float64),requires_grad=True).long()
    x_eval_tensor = torch.tensor(X_eval.astype(np.float32),requires_grad=True)
    s_eval_tensor = torch.tensor(S_eval.astype(np.float32),requires_grad=True)

    # Calculate disparities
    orig_scores = calc_disp(model, x_eval_tensor, y_eval_tensor, s_eval_tensor, 
                                                        [default_minority_group,default_majority_group])

    # Read in the GradientBoost model and apply it to the df_public
    knowLabel = 'withY'
    model_type = 'GradientBoost'

    for metric in metrics:
        
        # Read in the files with scores
        GT = pd.read_csv("%s/out/%s_%s_train_influences.csv"%(data_name, metric, data_name), index_col=0)
        BB_public = pd.read_csv("%s/out/%s_%s_%s_Wang_influences.csv"%(data_name, metric, data_name, 'public'), index_col=0) 
        BB_train = pd.read_csv("%s/out/%s_%s_%s_Wang_influences.csv"%(data_name, metric, data_name, 'train'), index_col=0) 
        
        # Read in a pre-trained model
        filename = '%s/models/%s_%s_%s_%s.sav'%(data_name, model_type, data_name, metric, knowLabel)
        ifs_predictor = pickle.load(open(filename, 'rb'))
        
        # Apply it to the public subset
        df = df_public.copy()
        df[group_column_name] = [0 if x == default_minority_group else 1 for x in df[group_column_name]]
        GT[group_column_name] = [0 if x == default_minority_group else 1 for x in GT[group_column_name]]
        BB_public[group_column_name] = [0 if x == default_minority_group else 1 for x in BB_public[group_column_name]]
        BB_train[group_column_name] = [0 if x == default_minority_group else 1 for x in BB_train[group_column_name]]

        # Prep this X - keep Y column (like train method)
        X_public = prepX(df, 'train', outcome_column_name=outcome_column_name, withInf = False)

        # Apply influence predictor
        pred = ifs_predictor.predict(X_public)

        # Add to original df
        df['influences'] = pred
        
        # Loop through each data frame 
        for label, title, temp_df in zip(['GT', 'BB_public', 'GB'],
                                         ['Ground Truth', 'BB Estimations on Public', 'GB Predictor on Public'], 
                                         [GT, BB_public, df]):
            
            fn = "%s/out/static_removal_%s_%s_%s_iterations.csv"%(data_name, data_name, metric, label)
            print("Removal: %s, %s, %s"%(data_name, metric, label))
            if keep_old and exists(fn):
                continue

            orig_disp = orig_scores.loc[default_majority_group,metric]- orig_scores.loc[default_minority_group,metric]
            orig_acc = orig_scores.loc['All','ACC'].flat[0]

            # Rank data by influence
            temp_df = temp_df.sort_values(by ='influences')
            short_df = temp_df[temp_df['influences']>0]

            accuracies = [orig_acc]
            disparities = [orig_disp]
            mean_infs = [0]
            sum_infs = [0]
            n_points_add = [0]
            n_points = 0
            sum_inf = 0
            mean_inf = 0

            # Keep add more and more positive points
            for i in range(20,len(short_df),20):
                # grab the top i points
                mean_inf = temp_df['influences'].tail(i).mean()
                sum_inf = sum_inf + sum(temp_df['influences'].tail(i))
                temp = temp_df.tail(i).drop(['influences'], axis=1)
                                
                # Find the like points 
                points_matched = temp[temp.apply(tuple,1).isin(df_train.apply(tuple,1))].shape[0]
                n_points = points_matched
                temp = df_train[~df_train.apply(tuple,1).isin(temp.apply(tuple,1))]

                # Re-prep train for model
                X_temp, Y_temp, S_temp = df_to_XYS(temp,
                                                      outcome_column_name = outcome_column_name,
                                                      group_column_name = group_column_name,
                                                      minority_group = default_minority_group)

                temp_ds = CurrentDataset(X_temp, Y_temp)

                temploader = torch.utils.data.DataLoader(temp_ds, batch_size=100,
                                                                          shuffle=True, num_workers=0)

                # Create model
                input_dim = temploader.dataset.__getitem__(0)[0].shape[0]
                y_model = LogisticRegression(input_dim)
                y_model.cpu()
                train(temploader, y_model)

                # Calculate disparities
                disp_scores = calc_disp(y_model, x_eval_tensor, y_eval_tensor, s_eval_tensor, 
                                                        [default_minority_group,default_majority_group])
                disparity = disp_scores.loc[default_majority_group,metric]- disp_scores.loc[default_minority_group,metric]
                disparities.append(disparity)
                accuracies.append(disp_scores.loc['All','ACC'].flat[0])
                mean_infs.append(mean_inf)
                sum_infs.append(sum_inf)
                n_points_add.append(n_points)
                if abs(disparity) < 0.01:
                    break
                elif disparities[-2] > 0 and disparity < 0:
                    break
                elif disparities[-2] < 0 and disparity > 0:
                    break
                
            d = {'N added':n_points_add,'Disparity':disparities, 'Accuracy':accuracies, 'Mean Influences':mean_infs, 'Sum Influences':sum_infs}
            disp_tb = pd.DataFrame(d)
            disp_tb.to_csv("%s/out/static_removal_%s_%s_%s_iterations.csv"%(data_name, data_name, metric, label))

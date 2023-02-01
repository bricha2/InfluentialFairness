# Read in libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import pickle
from kneed import KneeLocator

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

keep_old = True

def getKnee(expandingMean, curve, direction, S=75):
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
            print("Concave was 1 and convex was less, so knee will be 20 or less.")
            print(expandingMean)
            topn = 20 if len(expandingMean) >= 20 else len(expandingMean)
        else:
            print("Concave was 1 and convex were None and expanding mean was empty, so breaking.")
            return None
    elif kl_prim_knee == None:
        if kl_sec_knee != None:
            print("Concave was None and convex was not.")
            print(expandingMean)
            topn = kl_sec_knee
        elif len(expandingMean) > 0:
            print("Both concave and convex were None, just grabbing the top 20 or less.")
            print(expandingMean)
            topn = 20 if len(expandingMean) >= 20 else len(expandingMean)
        else:
            print("Both concave and convex were None and expanding mean was empty, so breaking.")
            return None
    else:
        topn = kl_prim_knee
    return topn

if __name__ == "__main__":
    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    data_name = config.get('SETTINGS','data_name')
    outcome_column_name = config.get('SETTINGS','outcome_column_name')
    group_column_name = config.get('SETTINGS','group_column_name')
    default_minority_group = config.get('SETTINGS','default_minority_group')
    default_majority_group = config.get('SETTINGS','default_majority_group')
            
    metrics = ['FDR','SP','EO', 'FPR']

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

        
        # Read in the correct model
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
        
        #step_sizes = ['knee',0.1,0.25,0.5,1]
        step_sizes = ['knee']
            
        for step_size in step_sizes:
        
            # Loop through each data frame 
            for name, title, inf_df in zip(['GT', 'BB_public', 'GB'],
                                           ['Ground Truth', 'BB Estimations on Public', 'GB Predictor on Public'], 
                                           [GT, BB_public, df]):
                print(name, metric, data_name)

                fn = "%s/out/nonstatic_duplication_%s_%s_%s_%s_iterations.csv"%(data_name, data_name, metric, 
                                                                                          name,str(step_size))
                if keep_old and exists(fn):
                    continue

                orig_disp = orig_scores.loc[default_majority_group,metric]- orig_scores.loc[default_minority_group,metric]
                orig_acc = orig_scores.loc['All','ACC'].flat[0]

                # Rank data by influence
                inf_df = inf_df.sort_values(by ='influences')
                short_df = inf_df[inf_df['influences']<0]

                accuracies = [orig_acc]
                disparities = [orig_disp]
                mean_infs = [0]
                n_points_add = [0]
                n_points = 0
                            
                not_df_train = df_train.copy()

                # Only duplicate until the size of df_train has doubled
                original_size = not_df_train.shape[0]
                while (n_points < original_size*2):
                    if len(short_df) == 0:
                        print("No points, breaking.")
                        break
                    if step_size == 'knee':
                        expandingMean = short_df['influences'].expanding().mean()
                        if len(expandingMean) == 1:
                            break
                        topn = getKnee(expandingMean, "concave", "increasing", S=75)
                        if topn == None:
                            break
                    else:
                        # Define topn based off of step_size
                        topn = int(short_df.shape[0]*step_size)
                                               
                    print("Top n: ", topn)
                        
                    # grab the top n points
                    mean_inf = inf_df['influences'].head(topn).mean()
                    topn_samples = inf_df.head(topn).drop(['influences'], axis=1)
                    
                    # Find the like points 
                    points_matched = topn_samples[topn_samples.apply(tuple,1).isin(not_df_train.apply(tuple,1))].shape[0]
                    n_points = n_points + points_matched
                    temp = topn_samples[topn_samples.apply(tuple,1).isin(not_df_train.apply(tuple,1))]
                    
                    # Add those top i points to df_train
                    temp_train = pd.concat([not_df_train, temp])

                    # Re-prep train for model
                    X_temp, Y_temp, S_temp = df_to_XYS(temp_train,
                                                          outcome_column_name = outcome_column_name,
                                                          group_column_name = group_column_name,
                                                          minority_group = default_minority_group)

                    temp_ds = CurrentDataset(X_temp, Y_temp)

                    temploader = torch.utils.data.DataLoader(temp_ds, batch_size=100,
                                                                              shuffle=True, num_workers=0)
                    

                    # Create tensors for subsequent steps
                    y_temp_tensor = torch.tensor(Y_temp.astype(np.float64),requires_grad=True).long()
                    x_temp_tensor = torch.tensor(X_temp.astype(np.float32),requires_grad=True)
                    s_temp_tensor = torch.tensor(S_temp.astype(np.float32),requires_grad=True)

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
                    n_points_add.append(n_points)
                    mean_infs.append(mean_inf)
                    if abs(disparity) < 0.01:
                        print("disparity low!")
                        break
                        

                    # Set df_train equal to temp train
                    not_df_train = temp_train.copy()

                    if 'GT' in name or 'GB' in name:
                        # First, get the gradient of loss of the train set wrt model parameters
                        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
                        print("f")
                        loss_fxn = nn.BCELoss()
                        grad_loss_train = grad_group(x_temp_tensor, y_temp_tensor, y_model, loss_fxn)

                        # Next, calculate the hessian wrt the gradient of loss for the training data
                        hess = hessian(grad_loss_train, y_model)

                        # Next, calculate the gradient of the fairness metric wrt to model parameters
                        # Calculate gradient of fairness metric on test set
                        grad_fair_test = grad_disp(y_model, metric, x_eval_tensor, y_eval_tensor, s_eval_tensor)

                        # Next, calculate gradient for each train point
                        influences = np.array([])
                        print("Calculating training influences...")
                        for i in tqdm(range(x_temp_tensor.shape[0])):
                            attr = x_temp_tensor[i]
                            label = y_temp_tensor[i]
                            grad_loss_ind = grad_individual(attr,label,y_model,loss_fxn)
                            grad_loss_ind = torch.vstack([g.view(-1,1) for g in grad_loss_ind])
                            try:
                                term2 = np.linalg.solve(hess.detach().numpy(), grad_loss_ind.detach().numpy())
                            except:
                                """print("Xtrain: ", X_train)
                                print("attr: ", attr)
                                print("label: ", label)
                                print("grad loss ind: ", grad_loss_ind.detach().numpy())
                                print("hess: ", hess.detach().numpy())"""
                                print("Broke.")
                                break
                            grad_fair_test = torch.vstack([g.view(-1,1) for g in grad_fair_test])
                            influence = - torch.matmul(grad_fair_test.T, torch.tensor(term2))
                            influences = np.append(influences, influence.detach().numpy().squeeze())

                        print("Original inf_df: ", inf_df.shape)

                        # If you need to use the GB predictor
                        if 'GB' in name:

                            # Remove top n points from inf_df
                            inf_df = inf_df.head(-topn).drop(['influences'], axis=1)
                            print("In GB predictor, after removing top n inf_df: ", inf_df.shape)
                            
                            if len(inf_df) == 0:
                                break

                            # Add influences to temp train
                            temp_train['influences'] = influences

                            # Train predictor with temp train
                            X = temp_train.drop(['influences'], axis = 1)
                            X[group_column_name] = [0 if x == default_minority_group else 1 for x in X[group_column_name]]           
                            y = temp_train['influences']
                            ifs_predictor.fit(X,y)
                            X_public = prepX(inf_df, 'train', outcome_column_name=outcome_column_name, withInf = False)

                            # Apply influence predictor
                            print("Shape of X_public: ", X_public.shape)
                            influences = ifs_predictor.predict(X_public)

                            # Drop influences from temp train
                            temp_train = temp_train.drop(['influences'], axis=1)
                        else:
                            inf_df = temp_train.copy()

                        # Prepare for next loop
                        inf_df['influences'] = influences
                        inf_df = inf_df.sort_values(by ='influences')
                        short_df = inf_df[inf_df['influences']<0]
                        print("Short df shape: ", short_df.shape)
                        print("N points: ", n_points)

                    else:

                        # Redefine test subset
                        inf_df = inf_df.head(-topn).drop(['influences'], axis=1)
                        
                        if len(inf_df) == 0:
                            break

                        # Re-prep train for model
                        inf_df[group_column_name] = [default_minority_group if x == 0 else default_majority_group 
                                                     for x in inf_df[group_column_name]]
                        X_test, Y_test, S_test = df_to_XYS(inf_df,
                                              outcome_column_name = outcome_column_name,
                                              group_column_name = group_column_name,
                                              minority_group = default_minority_group)
                        print(inf_df.head(10))
                        inf_df[group_column_name] = [0 if x == default_minority_group else 1 for x in inf_df[group_column_name]]

                        # Next, we calculate the Black-box influence scores
                        predict_yhat = lambda x: y_model(x,1)
                        predict_yhatd = lambda x: y_model(x)

                        # Get rows where sensitive attr is the minority group
                        X_test_min = X_test[S_test == 0,:]
                        Y_test_min = Y_test[S_test == 0] 
                        S_test_min = S_test[S_test == 0]
                        
                        if X_test_min.shape[0] == 0:
                            break

                        test_ds_min = CurrentDataset(X_test_min, Y_test_min)
                        testloader_min = torch.utils.data.DataLoader(test_ds_min, batch_size=100, shuffle=True, num_workers=0)

                        # Create true model
                        input_dim = testloader_min.dataset.__getitem__(0)[0].shape[0]
                        model_min = LogisticRegression(input_dim)
                        model_min.cpu()
                        train(testloader_min, model_min)
                        predict_ytrue = lambda x: model_min(x)

                        disparity = disp_scores.loc[default_minority_group,metric]-disp_scores.loc[default_majority_group,metric]

                        influences = getBlackBoxScores(torch.from_numpy(X_test_min.astype(int)), 
                                                       torch.from_numpy(X_test.astype(int)), 
                                                       predict_yhat, predict_ytrue, metric, disparity)

                        # Save influences to dataframe
                        inf_df['influences'] = influences

                        # Prepare for next loop
                        inf_df = inf_df.sort_values(by ='influences')
                        short_df = inf_df[inf_df['influences']<0]

                d = {'N added':n_points_add,'Disparity':disparities, 'Accuracy':accuracies, 'Mean Influences':mean_infs}
                disp_tb = pd.DataFrame(d)
                disp_tb.to_csv("%s/out/nonstatic_duplication_%s_%s_%s_%s_iterations.csv"%(data_name, data_name, metric, 
                                                                                          name,str(step_size)))

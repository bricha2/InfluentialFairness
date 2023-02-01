# This file runs the first three steps
import sys
import configparser
from influence_functions import *

# Code parameters
random_seed = 5968
train_size = 0.40
public_size = 0.30
eval_size = 0.30

printFiles = True
createModel = True
calcWhiteBox = True
calcBlackBox = True

if __name__ == '__main__':
    # Settings
    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    data_name = config.get('SETTINGS','data_name')
    outcome_column_name = config.get('SETTINGS','outcome_column_name')
    group_column_name = config.get('SETTINGS','group_column_name')
    default_minority_group = config.get('SETTINGS','default_minority_group')
    default_majority_group = config.get('SETTINGS','default_majority_group')
    
    random_state = np.random.RandomState(random_seed)
            
    data_file_name = 'data/%s_proxy_data.csv' % (data_name)

    # load dataset
    if (printFiles):
        raw_df = pd.read_csv(data_file_name)
        
        # Drop the first column
        raw_df = raw_df.iloc[:,1:]
        
        #if data_name != 'adult':
        raw_df[group_column_name] = [default_minority_group if x == 0 else default_majority_group 
                                     for x in raw_df[group_column_name]]

        # returns order train, audit, test
        df_train, df_eval, df_public = train_test_audit_split(df = raw_df,
                                                            train_size = train_size,
                                                            test_size = public_size,
                                                            audit_size = eval_size,
                                                            group_column_name = group_column_name,
                                                            outcome_column_name = outcome_column_name,
                                                            random_state = random_state)
        
        df_train.to_csv("%s/out/%s_train.csv"%(data_name, data_name))
        df_eval.to_csv("%s/out/%s_eval.csv"%(data_name, data_name))
        df_public.to_csv("%s/out/%s_public.csv"%(data_name, data_name))
                
    else:
        df_train = pd.read_csv("%s/out/%s_train.csv"%(data_name, data_name), index_col = 0)
        df_eval = pd.read_csv("%s/out/%s_eval.csv"%(data_name, data_name), index_col = 0)
        df_public = pd.read_csv("%s/out/%s_public.csv"%(data_name, data_name), index_col = 0)
            

    X_train, Y_train, S_train = df_to_XYS(df_train,
                                        outcome_column_name = outcome_column_name,
                                        group_column_name = group_column_name,
                                        minority_group = default_minority_group)
    
    X_eval, Y_eval, S_eval = df_to_XYS(df_eval,
                                    outcome_column_name = outcome_column_name,
                                    group_column_name = group_column_name,
                                    minority_group = default_minority_group)
            
    # creating train and valid data objects
    train_ds = CurrentDataset(X_train, Y_train)
    eval_ds = CurrentDataset(X_eval, Y_eval)
    
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=100,
                                                shuffle=True, num_workers=0)
    evalloader = torch.utils.data.DataLoader(eval_ds, batch_size=100,
                                                shuffle=False, num_workers=0)
    

    del train_ds, eval_ds

    # Create model
    input_dim = trainloader.dataset.__getitem__(0)[0].shape[0]
        
    if printFiles or createModel:
        model = LogisticRegression(input_dim)
        model.cpu()
        train(trainloader, model)
        save_model(model, '%s/%s_model.pth'%(data_name, data_name))
    else:
        model = load_model(input_dim, '%s/%s_model.pth'%(data_name, data_name))

    # Get results on test set
    #test(evalloader, model)

    # Make tensor of data
    y_train_tensor = torch.tensor(Y_train.astype(np.float64),requires_grad=True).long()
    x_train_tensor = torch.tensor(X_train.astype(np.float32),requires_grad=True)
    s_train_tensor = torch.tensor(S_train.astype(np.float32),requires_grad=True)
    y_eval_tensor = torch.tensor(Y_eval.astype(np.float64),requires_grad=True).long()
    x_eval_tensor = torch.tensor(X_eval.astype(np.float32),requires_grad=True)
    s_eval_tensor = torch.tensor(S_eval.astype(np.float32),requires_grad=True)

    # Calculate disparities
    if printFiles or createModel:
        disp_scores = calc_disp(model, x_train_tensor, y_train_tensor, s_train_tensor, 
                                [default_minority_group,default_majority_group])
        disp_scores.to_csv("%s/out/%s_train_disparities.csv"%(data_name, data_name))
        disp_scores = calc_disp(model, x_eval_tensor, y_eval_tensor, s_eval_tensor, 
                                [default_minority_group,default_majority_group])
        disp_scores.to_csv("%s/out/%s_eval_disparities.csv"%(data_name, data_name))
        
    # If we are calculating WB scores
    if calcWhiteBox:

        # First, get the gradient of loss of the train set wrt model parameters
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        loss_fxn = nn.BCELoss()
        grad_loss_train = grad_group(x_train_tensor, y_train_tensor, model, loss_fxn)

        # Next, calculate the hessian wrt the gradient of loss for the training data
        hess = hessian(grad_loss_train, model)

        # Next, calculate the gradient of the fairness metric wrt to model parameters
        metrics = ['SP','EO','TPR', 'TNR', 'FNR', 'FPR', 'FDR', 'FOR']
        for metric in metrics:
            # Calculate gradient of fairness metric on test set
            grad_fair_test = grad_disp(model, metric, x_eval_tensor, y_eval_tensor, s_eval_tensor)

            # Next, calculate gradient for each train point
            influences = np.array([])
            print("Calculating training influences...")
            for i in tqdm(range(x_train_tensor.shape[0])):
                attr = x_train_tensor[i]
                label = y_train_tensor[i]
                grad_loss_ind = grad_individual(attr,label,model,loss_fxn)
                grad_loss_ind = torch.vstack([g.view(-1,1) for g in grad_loss_ind])
                term2 = np.linalg.solve(hess.detach().numpy(), grad_loss_ind.detach().numpy())
                grad_fair_test = torch.vstack([g.view(-1,1) for g in grad_fair_test])
                influence = - torch.matmul(grad_fair_test.T, torch.tensor(term2))
                influences = np.append(influences, influence.detach().numpy().squeeze())

            # Save influences to a file
            out = df_train.copy()
            out['influences'] = influences
            out.sort_values(by='influences', key=abs, inplace=True)
            out.to_csv("%s/out/%s_%s_train_influences.csv"%(data_name, metric, data_name))
            
            out['prediction'] = model(x_train_tensor,1).detach().numpy()
            out.to_csv("%s/out/%s_%s_train_influences_withYhat.csv"%(data_name, metric, data_name))
        
        
    # If we are calculating BB scores
    if calcBlackBox:
        
        X_public, Y_public, S_public = df_to_XYS(df_public,
                                        outcome_column_name = outcome_column_name,
                                        group_column_name = group_column_name,
                                        minority_group = default_minority_group)
        
        x_public_tensor = torch.tensor(X_public.astype(np.float32),requires_grad=True)
        
        disp_scores = calc_disp(model, x_eval_tensor, y_eval_tensor, s_eval_tensor, 
                                [default_minority_group,default_majority_group])
        
        calcBlackBoxFn(df_public, df_public, model, disp_scores, default_minority_group, default_majority_group, 'public', 
                 data_name, outcome_column_name, group_column_name, x_apply_tensor = x_public_tensor)
        
        calcBlackBoxFn(df_public, df_train, model, disp_scores, default_minority_group, default_majority_group, 'train', 
                 data_name, outcome_column_name, group_column_name, x_apply_tensor = x_train_tensor)
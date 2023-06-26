'''



Important notes: 

sklearn.utils.Bunch: just an extended dictionary that allows attributes to referenced
by  key, bunch["value_key"], or by an attribute, bunch.value_key


Notes: 
R^2 number: lm2.score(X, y)




'''
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets as datasets




def dataset_df(dataset_name,
              verbose = False,
              target_name="target",
              dropna=True):
    load_dataset_func = getattr(datasets,f"load_{dataset_name}")
    
    if verbose:
        print(f"load_dataset_func = {load_dataset_func}")
    
    #this actually returns an sklearn utils.Bunch
    data = load_dataset_func()
    
    try:
        curr_data = data.data
        feature_names = data.feature_names
        targets = data.target
    except:
        curr_data = data["data"]
        feature_names = data["feature_names"]
        targets = data["target"] 
        
    df = pd.DataFrame(curr_data, columns=feature_names)
    df[target_name] = targets
    
    if dropna:
        df =df.dropna()
    return df

def load_boston():
    """
    MEDV: the median value of home prices
    
    """
    return dataset_df("boston",
              verbose = False,
              target_name="MEDV")


def MSE(y_true,y_pred=None,model=None,X = None,clf=None):
    """
    Purpose: Will calculate the MSE of a model
    
    """
    if model is None:
        model = clf
    if y_pred is None:
        y_pred = clf.predict(X)
    
    return mean_squared_error(y_true, y_pred)

def logistic_log_loss(clf,X,y_true):
    """
    Computes the Log loss, aka logistic loss or cross-entropy loss.
    on a model
    """
    y_pred = clf.predict_proba(X)
    return log_loss(y_true, y_pred)
    

def accuracy(clf,X,y):
    """
    Returns the accuracy of a classifier
    
    """
#     try:
#         X = X.to_numpy()
#     except:
#         pass
    
#     try:
#         y = y.to_numpy()
#     except:
#         pass
    
    
    return clf.score(X, y)

def train_val_test_split(
    X,
    y,
    test_size = 0.2,
    val_size = None,
    verbose = False,
    random_state = None, #can pass int to get reproducable results
    return_dict = False,
    ):
    """
    Purpose: To split the data into 
    1) train
    2) validation (if requested)
    3) test

    Note: All percentages are specified as number 0 - 1
    Process: 
    1) Split the data into test and train percentages
    2) If validation is requested, split the train into train,val
    by the following formula

    val_perc/ ( 1 - test_perc)  =  val_perc_adjusted
 
    3) Return the different splits
    
    
    Example: 
    (X_train,
     X_val,
     X_test,
     y_train,
     y_val,
     y_test) = sklu.train_val_test_split(
        X,
        y,
        test_size = 0.2,
        val_size = 0.2,
        verbose = True)
    """
    train_size = None


    X_train, X_test, y_train, y_test  = train_test_split(
                                            X,
                                            y,
                                            test_size = test_size,
                                            random_state = random_state)
    if val_size is None:
        if verbose:
            print(f"For Train/Val/Test split of {train_size}/{test_size}"
                  f" = {len(X_train)}/{len(X_test)}")
        if return_dict:
            data_splits = dict(
            X_train=X_train,
             X_test=X_test,
             y_train=y_train,
             y_test=y_test
            )
            return data_splits
        else:
            return X_train, X_test, y_train, y_test

    
    val_size_adj = val_size/(1 - test_size)
    
    X_train, X_val, y_train, y_val  = train_test_split(
                                            X_train,
                                            y_train,
                                            test_size = val_size_adj,
                                            random_state = random_state)
    
    if verbose:
        print(f"For Train/Val/Test split of {train_size}/{val_size}/{test_size}"
              f" = {len(X_train)}/{len(X_val)}/{len(X_test)}")
    if return_dict:
        data_splits = dict(
        X_train=X_train,
         X_val=X_val,
         X_test=X_test,
         y_train=y_train,
         y_val=y_val,
         y_test=y_test
        )
        return data_splits
    else:
        return X_train,X_val,X_test,y_train,y_val,y_test


def k_fold_df_split(
    X,
    y,
    target_name = None,
    n_splits = 5,
    random_state = None,
    ):
    """
    Purpose: 
    To divide a test and training dataframe
    into multiple test/train dataframes to use for k fold cross validation

    Ex: 
    n_splits = 5
    fold_dfs = sklu.k_fold_df_split(
        X_train_val,
        y_train_val,
        n_splits = n_splits)

    fold_dfs[1]["X_train"]
    """
    if y is None:
        X,y = pdml.X_y(X,target_name)

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=False)

    folds_test_train = dict()
    for j,(train_index, test_index) in enumerate(kf.split(X)):
        #print("TRAIN:", train_index, "TEST:", test_index)

        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        folds_test_train[j] = dict(X_train=X_train_fold,
                                  X_test = X_test_fold,
                                  y_train = y_train_fold,
                                  y_test = y_test_fold)

    return folds_test_train

def optimal_parameter_from_kfold_df(
    df,
    parameter_name = "k",
    columns_prefix = "mse_fold",
    higher_param_higher_complexity = True,
    d = True,
    verbose = False,
    return_df = False,
    standard_error_buffer = False,
    plot_loss = True,
    **kwargs, #mostly arguments for plotting
                                 ):
    """
    Purpose: Will find the optimal parameter 
    based on a dataframe of the mse scores for different parameters
    
    Ex: 
    opt_k,ret_df = sklu.optimal_parameter_from_mse_df(
    best_subset_df,
    parameter_name = "k",
    columns_prefix = "mse_fold",
    higher_param_higher_complexity = True,
    standard_error_buffer = True,
    verbose = True,
    return_df = True
                                 )
ret_df
    """
    best_subset_df= df
    
    mse_col = [k for k in best_subset_df.columns if (columns_prefix in k) and 
          ("mean" not in k) and ("std_dev" not in k) and ("std_err" not in k)]
    
    mean_col = f"{columns_prefix}_mean"
    std_error_col = f"{columns_prefix}_std_err"
    
    best_subset_df[mean_col] = best_subset_df[mse_col].mean(axis=1)
    #best_subset_df[f"{columns_prefix}_std_dev"] = best_subset_df[mse_col].std(axis=1)
    best_subset_df[std_error_col] = best_subset_df[mse_col].sem(axis=1)
    
    curr_data = best_subset_df.query(f"{mean_col} == {best_subset_df[mean_col].min()}"
                                    )[[mean_col,std_error_col]].to_numpy()
    
    mean_opt, std_err_opt=  curr_data[0]
    
    if not standard_error_buffer:
        std_err_opt = 0
        
    if higher_param_higher_complexity:
        optimal_k = best_subset_df.query(f"{mean_col} <= {mean_opt + std_err_opt}")[parameter_name].min()
    else:
        optimal_k = best_subset_df.query(f"{mean_col} <= {mean_opt + std_err_opt}")[parameter_name].max()
        
    if verbose:
        print(f"mean_opt= {mean_opt}",f"std_err_opt = {std_err_opt}",f"{columns_prefix} cutoff = {mean_opt + std_err_opt}")
        print(f"optimal_{parameter_name} = {optimal_k}")
    
    
    if plot_loss:
        from machine_learning_tools import pandas_ml as pdml
        pdml.plot_df_x_y_with_std_err(
        best_subset_df,
        x_column= parameter_name,
        y_column = mean_col,
        std_err_column = std_error_col,
        **kwargs
        )
    
    if return_df:
        return optimal_k,best_subset_df
    else:
        return optimal_k

    
    
def random_regression_with_informative_features(
    n_samples=306,
    n_features=8000,
    n_informative=50,   
    random_state=42,
    noise=0.1,
    return_true_coef = True,
    ):

    """
    Purpose: will create a random regression
    with a certain number of informative features
    
    """
    X, y, coef = make_regression(n_samples=n_samples,
                                 n_features=n_features, 
                                 n_informative=n_informative,
                                noise=noise,
                                 shuffle=True,
                                 coef=True,
                                 random_state=random_state)

    X /= np.sum(X ** 2, axis=0)  # scale features
    if return_true_coef:
        return X,y,coef
    else:
        return X,y
    
def CV_optimal_param_1D(
    parameter_options,
    clf_function,
    
    #arguments for loss function
    loss_function,
    
    #cross validation parameters
    n_splits = 5,
    
    # parameters for splits
    data_splits = None,
    X = None,
    y = None,
    test_size =0.2,
    val_size = 0.2,

    #parameters for the type of classifier
    
    clf_parameters = dict( ),

    
    
    
    #arguments for the determination of the optimal parameter
    standard_error_buffer = True,
    plot_loss = False,
    
    #arguments for return
    return_data_splits = False,

    verbose = False,
    ):

    """
    Purpose: To Run Cross Validation by Hand with Specific
    - Dataset
    - Model Type
    - 1D Parameter Grid to Search over
    - Loss function to measure
    - Method of evaluating the best loss function

    Pseudocode: 
    0) Define the parameter space to iterate over
    1) Split the Data into,test,training and validation
    2) Combine the validation and training datasets
    in order to do cross validation
    3) Compute the datasets for each cross validation 

    For every parameter option:
        For every K fold dataset:
            Train the model on the dataset
            Measure the MSE or another loss for that model
            Store the certain loss
        Find the average loss and the standard error on the loss

    Pick the optimal parameter by one of the options:
    a) Picking the parameter with the lowest average loss
    b) Picking the parameter value that is the least complex model
     that is within one standard deviation of the parameter with the
     minimum average loss
     
     Example: 
     clf,data_splits = sklu.CV_optimal_param_1D(
        parameter_options = dict(C = np.array([10.**(k) for k in np.linspace(-4,3,25)])),

        X = X,
        y = y,

        #parameters for the type of classifier
        clf_function = linear_model.LogisticRegression,
        clf_parameters = dict(
            penalty = "l1",
             solver="saga",
             max_iter=10000, ),

        #arguments for loss function
        loss_function = sklu.logistic_log_loss,

        #arguments for the determination of the optimal parameter
        standard_error_buffer = True,
        plot_loss = True,


        #arguments for return
        return_data_splits = True,

        verbose = True,
        )

    """

    if data_splits is None:
        if X is None or y is None:
            raise Exception("X and y must be set if data_splits is None")
        
        (X_train,
         X_val,
         X_test,
         y_train,
         y_val,
         y_test) = sklu.train_val_test_split(
            X,
            y,
            test_size = test_size,
            val_size = val_size,
            verbose = verbose)

        data_splits = dict(
        X_train=X_train,
         X_val=X_val,
         X_test=X_test,
         y_train=y_train,
         y_val=y_val,
         y_test=y_test
        )

    if "X_val" in data_splits.items():
        X_train = data_splits["X_train"]
        X_val = data_splits["X_val"]
        X_test = data_splits["X_test"]

        y_train = data_splits["y_train"]
        y_val = data_splits["y_val"]
        y_test = data_splits["y_test"]

        X_train_val = pd.concat([X_train,X_val])
        y_train_val = pd.concat([y_train,y_val])

    else:
        X_train_val = data_splits["X_train"]
        X_test = data_splits["X_test"]

        y_train_val = data_splits["y_train"]
        y_test = data_splits["y_test"]


    fold_dfs = sklu.k_fold_df_split(
        X_train_val,
        y_train_val,
        n_splits = n_splits)

    cv_dicts = []
    for param_name,C_options in parameter_options.items():
        for c in tqdm(C_options):
            curr_dict = {param_name:c}
            for fold_idx,fold_data in fold_dfs.items():

                #setting classifier type
                p_dict = clf_parameters.copy()
                p_dict.update({param_name:c})
                clf = clf_function(**p_dict)

                #training the classifier
                clf.fit(fold_data["X_train"],
                       fold_data["y_train"])

                #computes the loss for the classifier
                loss = loss_function(clf = clf,
                                    X = fold_data["X_test"],
                                    y_true = fold_data["y_test"],)

                curr_dict[f"{loss_function.__name__}_fold_{fold_idx}"] = loss


            cv_dicts.append(curr_dict)

    # compiles the results
    df = pd.DataFrame.from_records(cv_dicts)



    # best parameter using the 1 Standard Error Trick
    opt_C, kfold_stat_df = sklu.optimal_parameter_from_kfold_df(
        df,
        parameter_name = param_name,
        columns_prefix = loss_function.__name__,
        return_df = True,
        standard_error_buffer = standard_error_buffer,
        verbose = verbose,
        plot_loss = plot_loss,

    )

    # train the final model on the optimal parameter:
    p_dict = clf_parameters.copy()
    p_dict.update({param_name:opt_C})
    clf = clf_function(**p_dict)
    clf.fit(X_train_val,y_train_val)

    # print the statistics of the optimal parameter
    if verbose:
        print(f"\n Cross Validation Statistics")
        print(f"{clf} Hand optimal C = {opt_C}")

        hand_opt_loss_val = loss_function(clf=clf,X=X_train_val,y_true=y_train_val)
        hand_opt_loss_test = loss_function(clf=clf,X=X_test,y_true=y_test)
        print(f"hand_opt_loss_val= {hand_opt_loss_val}")
        print(f"hand_opt_loss_test= {hand_opt_loss_test}\n")

    if return_data_splits:
        return clf,data_splits
    else:
        return clf


def accuracy_score(y_true,y_pred,**kwargs):
    return sklearn.metrics.accuracy_score(y_true,y_pred,**kwargs)




from . import sklearn_utils as sklu
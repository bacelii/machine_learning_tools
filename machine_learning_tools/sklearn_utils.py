import sklearn
import sklearn.datasets as datasets
import pandas as pd


"""
Important notes: 

sklearn.utils.Bunch: just an extended dictionary that allows attributes to referenced
by  key, bunch["value_key"], or by an attribute, bunch.value_key


Notes: 
R^2 number: lm2.score(X, y)

"""


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


from sklearn.metrics import mean_squared_error
def MSE(y_true,y_pred=None,model=None,X = None):
    """
    Purpose: Will calculate the MSE of a model
    
    """
    if y_pred is None:
        y_pred = model.predict(X)
    
    return mean_squared_error(y_true, y_pred)

def accuracy(X,y):
    """
    Returns the accuracy of a classifier
    
    """
    return clf.score(df_X, df_y)

from sklearn.model_selection import train_test_split
def train_val_test_split(
    X,
    y,
    test_size = 0.2,
    val_size = None,
    verbose = False,
    random_state = None, #can pass int to get reproducable results
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
    return X_train,X_val,X_test,y_train,y_val,y_test

import pandas as pd
from sklearn.model_selection import KFold 

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

def optimal_parameter_from_mse_kfold_df(
    df,
    parameter_name = "k",
    columns_prefix = "mse_fold",
    higher_param_higher_complexity = True,
    standard_error_buffer = True,
    verbose = False,
    return_df = False
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
        print(f"mean_opt= {mean_opt}",f"std_err_opt = {std_err_opt}",f"mse cutoff = {mean_opt + std_err_opt}")
        print(f"optimal_k= {optimal_k}")
    
    
    if return_df:
        return optimal_k,best_subset_df
    else:
        return optimal_k

    
    
from sklearn.datasets import make_regression
import numpy as np
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

import sklearn_utils as sklu
'''



Functions to help with featuer evaluation





'''
from sklearn.feature_selection import chi2, f_regression
import itertools
import numpy as np
import sklearn.feature_selection as fs

# ------ Evaluation of models ---------



all_evaluation_methods = ["R_squared","MSE"]

def best_subset_k(
    df,
    k,
    model,
    target_name=None,
    y = None,
    verbose = False,
    evaluation_method = "R_squared",#MSE
    return_model = False,
    
    ):

    """
    Purpose: To pick the best subset of features for a certain 
    number of features allowed 

    - evalutation of best is chosen by the 
    evaluation method: R^2, MSE

    Pseudocode: 
    0) divide df into X,y
    1) Get all choose k subsets of the features
    2) For each combination of features:
    - find the evaluation score
    
    sklm.best_subset_k(
    df,
    k = 2,
    target_name = target_name,
    model = sklm.LinearRegression(),
    evaluation_method = "MSE",
    verbose = True
    )

    """
    
    def choose_k_combinations(array,k):
        return list(itertools.combinations(array,k))

    if y is None:
        y = df[target_name]

    X = pml.df_no_target(df,target_name)

    features = list(X.columns)

    if verbose:
        print(f"features = {features}")

    all_feature_combinations = choose_k_combinations(features,k)

    if verbose:
        print(f"all_feature_combinations = {all_feature_combinations}")

    feat_comb_scores = []
    for feat_comb in all_feature_combinations:
        fit_model = model.fit(X[list(feat_comb)],y)
        if evaluation_method == "R_squared":
            curr_eval = fit_model.score(X[list(feat_comb)],y)
            argfunc = np.argmax
        elif evaluation_method == "MSE":
            curr_eval = sklu.MSE(y,model = fit_model,X = X[list(feat_comb)] )
            argfunc = np.argmin
        else: 
            raise Exception(f"Unimplemented evaluation_method = {evaluation_method}")

        feat_comb_scores.append(curr_eval)
        if verbose:
            print(f"feature_comb ({feat_comb}): {curr_eval}")

    winning_index = argfunc(feat_comb_scores)
    winning_features = all_feature_combinations[winning_index]

    if verbose:
        print(f"\nBest features for k = {k}: {winning_features} with {evaluation_method}"
              f" score {np.max(feat_comb_scores)}")
        
    if return_model:
        fit_model = model.fit(X[list(winning_features)],y)
        return winning_features,fit_model
    else:
        return winning_features

def best_subset_k_individual_sklearn(df,
                         target_name,
                         k,
                          model_type = "regression",
                          verbose = False,
                          return_data= False,
                         ):
    """
    Purpose: To run the sklearn best subsets k using 
    built in sklearn method (NOTE THIS SELECTS THE BEST FEATURES INDIVIDUALLY)
    
    Useful Link: https://www.datatechnotes.com/2021/02/seleckbest-feature-selection-example-in-python.html
    
    
    Example
    
    '''
    Note: This method uses a the following evaluation criteria for best feature
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html

    1) The correlation between each regressor and the target is computed, that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) * std(y)).
    2) It is converted to an F score then to a p-value.

    '''
    best_features_over_sk = []

    for k in tqdm(range(1,n_featuers+1)):
        eval_method = "sklearn"

        curr_best = fsu.best_subset_k_individual_sklearn(
        df,
        k = k,
        target_name = target_name,
        verbose = False,
        )
        best_features_over_sk.append(dict(k = k,
                                   evaluation_method = eval_method,
                                   best_subset = curr_best))

    import pandas as pd
    print(f"Using sklearn method")
    pd.DataFrame.from_records(best_features_over_sk)
    
    
    """
    
    if model_type == "regression":
        score_func = f_regression
    elif model_type == "classification":
        score_func = chi2
    else:
        raise Exception("")
    
    select = fs.SelectKBest(score_func=f_regression, k=k)
    
    X,y = pml.X_y(df,target_name)
    
    z = select.fit_transform(X, y)
    if verbose:
        print(f"After selecting best {k} features:", z.shape)
        
    
    column_idx = select.get_support()
    best_sub = np.array(list(X.columns))[column_idx]
    
    if verbose:
        print(f"Best subset: {best_sub}")
        
    if return_data:
        return best_sub,z
    else:
        return best_sub
    
def reverse_feature_elimination(df,
                                k,
                                model,
                                target_name=None,
                                y = None,
                               verbose = False,
                               return_model = False):
    """
    Use sklearn function for recursively
    elimininating the least important features
    
    How does it pick the best features? 
    the absolute value of the model.coef_ (not considering p_value)
    
    """
    
    if y is None:
        X,y = pml.X_y(df,target_name)
    else:
        X,y = df,y
    
    
    selector = fs.RFE(model,
          n_features_to_select = k,
          step = 1,
                     verbose=verbose)
    selector.fit(X,y)
    
    best_features = pml.feature_names(X)[selector.support_]
    if return_model:
        model.fit(X[list(best_features)],y)
        return best_features,model
    else:
        return best_features
    


#--- from machine_learning_tools ---
from . import pandas_ml as pml
from . import sklearn_models as sklm
from . import sklearn_utils as sklu

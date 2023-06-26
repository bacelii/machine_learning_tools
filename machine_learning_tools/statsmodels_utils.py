'''



Purpose: to export some functionality of the 
statsmodels library for things like

Application: Behaves a lot like the R functions

-linear regression



Notes: 
lm1.rsquared gives the r squared value



Notes:




'''
import numpy as np
import pandas as pd
import statsmodels.api as sm


def linear_regression(df,
                      target_name,
                      add_intercept = True,
                      print_summary = True):
    """
    Purpose: To run a linear regression and then
    print out the summary of the coefficients
    
    eX: 
    from machine_learning_tools import statsmodels_utils as smu
    smu.linear_regression(df_raw[[target_name,"LSTAT"]],
                     target_name = target_name,
                     add_intercept = True,
                     )
    
    """
    y = df[[k for k in df.columns if k != target_name]]
    
    if add_intercept:
        y = sm.add_constant(y)
    
    est = sm.OLS(df[target_name], y)
    est2 = est.fit()
    
    if print_summary:
        print(est2.summary())
        
    return est2
    
def coef(model):
    return model.params
def pvalues(model):
    return model.pvalues


def coef_pvalues_df(model):
    coef = smu.coef(model)
    pvals = smu.pvalues(model)

    df = pd.DataFrame.from_records([coef,pvals]).T
    df.columns = ["coef","pvalues"]
    return df

def ranked_features(model,
                    pval_max = 0.001,
                   verbose = False,
                   return_filtered_features = False):
    """
    Purpose: Will get the ranked features by their 
    coefficients and filter away tose with a a high p value
    """
    df = smu.coef_pvalues_df(model)
    
    filt_df = df.query(f"pvalues < {pval_max}")
    invalid_df = df.query(f"pvalues >= {pval_max}")
    high_p_featueres = list(invalid_df.index)
    
    if verbose:
        print(f"high_p_featueres= {high_p_featueres}")
        
    coeff_vals = filt_df["coef"].to_numpy()
    filt_df_names = np.array(list(filt_df.index))
    
    ordered_idx = np.flip(np.argsort(np.abs(coeff_vals)))
    ordered_features = filt_df_names[ordered_idx]
    
    if verbose:
        print(f"ordered_coef = {coeff_vals[ordered_idx]}")
        print(f"ordered_features = {ordered_features}")
        
    if return_filtered_features:
        return ordered_features,coeff_vals
    else:
        return ordered_features

    



from . import statsmodels_utils as smu
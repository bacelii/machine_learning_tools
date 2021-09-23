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

import sklearn_utils as sklu
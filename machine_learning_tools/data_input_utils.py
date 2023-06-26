
import numpy as np
import pandas as pd


def df_from_rda(filepath,verbose = True,
               reset_index= False,
               use_label_for_name = True):
    """
    conert an .rda r file into
    a pandas dataframe
    
    """
    
    import pyreadr
    result = pyreadr.read_r('./authors.rda') # also works for Rds, rda

    # done! let's see what we got
    # result is a dictionary where keys are the name of objects and the values python
    # objects
    curr_key = list(result.keys())[0]
    if verbose:
        print(f"Index = {curr_key}")
    df = result[curr_key]
    if reset_index:
        if use_label_for_name:
            df["label"] = df.index.values
        else:
            df[curr_key] = df.index.values
        df.index = np.arange(len(df.index.values))
    return df



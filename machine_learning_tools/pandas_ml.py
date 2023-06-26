'''



Purpose: pandas functions that are useful for machine learning


.iloc: indexes with integers
ex: df_test.iloc[:5] --> gets first 5 rows
.loc: indexes with strings
Ex: df_test.loc[df.columns,df.columns[:5]]



'''
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
    
    
def df_no_target(df,target_name):
    return df[[k for k in df.columns if k != target_name]]

def n_features(df,target_name=None):
    return len(pdml.df_no_target(df,target_name).columns)

def X_y(df,target_name):
    return pdml.df_no_target(df,target_name),df[target_name]

def df_from_X_y(
    X,
    y,
    data_column_names = "feature",
    target_name = "target"):
    """
    Ex: 
    pdml.df_from_X_y(X_trans,y,target_name = "cell_type")
    """
    
    f = pd.DataFrame(X)
    f.columns = [f"{data_column_names}_{i}" for i in range(X.shape[1])]
    f[target_name] = y
    
    return f
    

def feature_names(df,target_name=None):
    return np.array(list(pdml.df_no_target(df,target_name).columns))

def df_column_summaries(df):
    return df.describe()

def filter_away_nan_rows(df):
    return df[~(df.isna().any(axis=1))]

def dropna(axis=0):
    """
    More straight forward way for dropping nans
    """
    return df.dropna(axis)

def correlations_by_col(df,
                           correlation_method = "pearson"):
    """
    will return a table that has the correlations between
    all the columns in the dataframe
    
    other correlations methods: "pearson","spearman",'kendall'
    
    """
    return df.corr(correlation_method)

def correlations_to_target(df,
                           target_name = "target",
                            correlation_method = "pearson",
                           verbose = False,
                           sort_by_value = True,
                          ):
    """
    Purpose: Will find the correlation between all
    columns and the 
    """
    #1) gets the correlation matrix
    corr_df = pdml.correlations_by_col(df,
                                     correlation_method = correlation_method)
    
    #2) only gets the last row (with the target)
    corr_with_target = corr_df.loc[target_name][[k for k in df.columns if k != target_name]]
    
    if sort_by_value:
        corr_with_target= corr_with_target.sort_values(ascending=False)
    
    return corr_with_target


def df_mean(df):
    return df.mean()

def df_std_dev(df):
    return df.std()

def center_df(df):
    return df - df.mean()

def hstack(dfs):
    return pd.concat(dfs,axis = 1)

def split_df_by_target(df,target_name):
    return [x for _, x in df.groupby(target_name)]



# ========= pandas visualizations ==============
def plot_df_x_y_with_std_err(
    df,
    x_column,
    y_column=None,
    std_err_column=None,
    log_scale_x = True,
    log_scale_y = True,
    verbose = False
    ):
    """
    Purpose: to plot the x and y 
    columns where the y column has
    an associated standard error with it
    
    Example: 
    from machine_learning_tools import pandas_ml as pdml
    pdml.plot_df_x_y_with_std_err(
    df,
        x_column= "C",
    )
    """
    
    fig,ax = plt.subplots()

    if y_column is None:
        y_column = [k for k in df.columns if "mean" in k][0]

    if std_err_column is None:
        std_err_column = [k for k in df.columns if "std_err" in k][0]

    if verbose:
        print(f"Using std_err_column = {std_err_column}")

    df.plot(x_column,
            y_column,
            yerr = std_err_column,
            ax = ax)
    
    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    
    plt.show()
    
def csv_to_df(csv_filepath):
    return pd.read_csv(csv_filepath) 
    
def df_to_csv(df,
              output_filename="df.csv",
              output_folder = "./",
              file_suffix = ".csv",
            output_filepath = None,
             verbose = False,
             return_filepath = True,
              compression = "infer",
             index = True,):
    """
    Purpose: To export a dataframe as a csv
    file
    """
    if output_filepath is None:
        output_folder = Path(output_folder)
        output_filename = Path(output_filename)
        output_filepath = output_folder / output_filename

    output_filepath = Path(output_filepath)
    
    if str(output_filepath.suffix) != file_suffix:
            output_filepath = Path(str(output_filepath) + file_suffix)
    
    output_path = str(output_filepath.absolute())
    if verbose: 
        print(f"Output path: {output_path}")
        
    df.to_csv(str(output_filepath.absolute()), sep=',',index=index,compression=compression)
    
    if return_filepath:
        return output_path

def df_to_gzip(df,
              output_filename="df.gzip",
              output_folder = "./",
            output_filepath = None,
             verbose = False,
             return_filepath = True,
             index = False,):
    """
    Purpose: Save off a compressed version of dataframe
    (usually 1/3 of the size)
    
    """
    return df_to_csv(df,
              output_filename=output_filename,
              output_folder = output_folder,
              file_suffix = ".gzip",
            output_filepath = output_filepath,
             verbose = verbose,
             return_filepath = return_filepath,
              compression = "gzip",
             index = index,)

def gzip_to_df(filepath):
    return pd.read_csv(filepath,
                       compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)

    
    



from . import pandas_ml as pdml
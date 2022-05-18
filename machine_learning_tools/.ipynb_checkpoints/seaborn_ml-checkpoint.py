import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas_ml as pdml
import pandas as pd

figsize_default = (10,10)

def corrplot(df,
             figsize = figsize_default,
             fmt='.2f',
             annot = True,
             **kwargs):
    """
    Purpose: Computes and plots the correlation
    """
    fig,ax = plt.subplots(figsize=figsize)
    sns.heatmap(pdml.correlations_by_col(df), annot=annot, fmt=fmt,ax=ax)

def plot_with_param(plot_func,
                   width_inches=10,
                   height_inches = 10,
                   **kwargs):
    fix, ax = plt.subplots(figsize = (width_inches,height_inches))
    
    return plot_func(ax=ax,**kwargs)

def hist(array,
        bins=50,
        figsize = figsize_default,
        **kwargs):
    fig,ax = plt.subplots(figsize=figsize)
    return sns.distplot(array,bins = bins,ax = ax,**kwargs)

def scatter_2D(x,y,
               x_label="feature_1",
               y_label = "feature_2",
               title = None,
               alpha=0.5,**kwargs):
    data = pd.DataFrame({
    x_label: x,
    y_label: y,
    })
    joint_obj = sns.jointplot(x=x_label,
                         y=y_label,
                         data = data,
                         kind = "scatter",
                         joint_kws={'alpha':alpha},
                         **kwargs)
    if title is not None:
        joint_obj.ax_marg_x.set_title(f"{title}")
        
    return joint_obj
    
def pairplot(df,**kwargs):
    return sns.pairplot(df,**kwargs)

def hist2D(x_df,y_df,n_bins = 100,cbar=True,**kwargs):
    return sns.histplot(x=x_df,#.iloc[:1000], 
           y=y_df,#.iloc[:1000],
            bins=n_bins,
             cbar=True,
                 **kwargs
           )
    

from matplotlib.colors import LogNorm
def heatmap(array,
            cmap = sns.cm.rocket_r,
            logscale = True,
            title=None,
             ax = None,
            **kwargs):
    """
    Purpose: Will make a heatmap
    """
    if ax is None:
        fig,ax = plt.subplots(1,1)
    if logscale:
        sns.heatmap(array, square=True, norm=LogNorm(),ax=ax,cmap=cmap)
    else:
        sns.heatmap(array, square=True,ax=ax,cmap=cmap)
    
    if title is not None:
        ax.set_title(title)
    
    return ax

import matplotlib.pyplot as plt
def save_plot_as_png(
    sns_plot,
    filename = "seaborn_plot.png"
    ):
    
    
    fig = sns_plot.fig
    fig.savefig(filename) 
    
    
import numpy_utils as nu
import pandas_utils as pu
def pairwise_hist2D(
    df,
    reject_outliers = True,
    verbose = True):
    df_pair_plot = df
    columns = list(df_pair_plot.columns)
    for i,c1 in enumerate(df_pair_plot.columns):
        for j,c2 in enumerate(df_pair_plot.columns):
            if j > i:
                if verbose:
                    print(f"\n\n\n--- working on {c1} vs {c2}-----")
                df_pair_plot_no_nan = pu.filter_away_nan_rows(df_pair_plot[[c1,c2]])
                if verbose:
                    print(f"# of after nans filtered = {len(df_pair_plot_no_nan)}")
                x = df_pair_plot_no_nan[c1].to_numpy()
                y = df_pair_plot_no_nan[c2].to_numpy()
                if reject_outliers:
                    x_mask = nu.reject_outliers(x,return_mask = True)
                    y_mask = nu.reject_outliers(y,return_mask = True)
                    mask = np.logical_and(x_mask,y_mask)
                    if verbose:
                        print(f"# of datapoints after outlier = {np.sum(mask)}")
                    x = x[mask]
                    y = y[mask]

    #             fig,ax = plt.subplots(1,1)
    #             ax.scatter(x[mask],y[mask])
                ax = sml.hist2D(x,y)
                ax.set_xlabel(c1)
                ax.set_ylabel(c2)
                ax.set_title(f"{c2} vs {c1}")
                plt.show()
                #break
        #break

import seaborn_ml as sml
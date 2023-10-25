
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

def lineplot(
    df,
    x,
    y,
    hue=None,
    **kwargs):
    
    return sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        **kwargs)

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
    


"""
Arguments to set for heatmap:

cmap='Blues',
annot=True,
fmt=".0f",
annot_kws={
    'fontsize': 16,
    'fontweight': 'bold',
    'fontfamily': 'serif'
}
"""
def heatmap(df,
            cmap ="Blues",# sns.cm.rocket_r,
            annot=True,
            logscale = True,
            title=None,
            figsize=None,
            fontsize=16,
            axes_fontsize = 30,
             ax = None,
            fmt = None,
            **kwargs):
    """
    Purpose: Will make a heatmap
    """
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=figsize)
    if logscale:
        kwargs["norm"] = LogNorm()
    if fmt is not None:
        kwargs["fmt"] = fmt
    sns.heatmap(
        df,
        square=True,
        ax=ax,
        cmap=cmap,
        annot=annot,
        annot_kws={
            'fontsize': fontsize,
            'fontweight': 'bold',
            'fontfamily': 'serif'
        },
        **kwargs)
    
    if title is not None:
        ax.set_title(title)
        
    ax = mu.set_axes_font_size(ax,axes_fontsize)
    
    return ax

def save_plot_as_png(
    sns_plot,
    filename = "seaborn_plot.png"
    ):
    
    
    fig = sns_plot.fig
    fig.savefig(filename) 
    
    
def pairwise_hist2D(
    df,
    reject_outliers = True,
    percentile_upper = 99.5,
    percentile_lower = None,
    features = None,
    verbose = True,
    return_data = False,
    bins = "auto"):
    df_pair_plot = df
    
    if features is None:
        columns = list(df_pair_plot.columns)
    else:
        columns = features
        
    if return_data:
        data = dict()
    for i,c1 in enumerate(columns):
        #print(f"i = {i}")
        for j,c2 in enumerate(columns):
            #print(f"j = {j}")
            if j > i:
                if verbose:
                    print(f"\n\n\n--- working on {c1} vs {c2}-----")
                df_pair_plot_no_nan = pu.filter_away_nan_rows(df_pair_plot[[c1,c2]])
                if verbose:
                    print(f"# of after nans filtered = {len(df_pair_plot_no_nan)}")
                x = df_pair_plot_no_nan[c1].to_numpy().astype("float")
                y = df_pair_plot_no_nan[c2].to_numpy().astype("float")
                if reject_outliers:
                    x_mask = nu.reject_outliers(x,return_mask = True)
                    y_mask = nu.reject_outliers(y,return_mask = True)
                    mask = np.logical_and(x_mask,y_mask)
                    if verbose:
                        print(f"# of datapoints after outlier = {np.sum(mask)}")
                    x = x[mask]
                    y = y[mask]
                    
                if percentile_upper is not None:
                    x_mask = x <= np.percentile(x,percentile_upper)
                    y_mask = y <= np.percentile(y,percentile_upper)
                    mask = np.logical_and(x_mask,y_mask)
                    
                    if verbose:
                        print(f"# of datapoints after percentile_upper"
                              f" ({percentile_upper}) = {np.sum(mask)}")
                    x = x[mask]
                    y = y[mask]
                    
                if percentile_lower is not None:
                    x_mask = x >= np.percentile(x,percentile_lower)
                    y_mask = y >= np.percentile(y,percentile_lower)
                    mask = np.logical_and(x_mask,y_mask)
                    
                    if verbose:
                        print(f"# of datapoints after percentile_upper"
                              f" ({percentile_upper}) = {np.sum(mask)}")
                    x = x[mask]
                    y = y[mask]
                    

    #             fig,ax = plt.subplots(1,1)
    #             ax.scatter(x[mask],y[mask])
                
                if verbose:
                    print(f"corr = {stu.corr(x,y)}, corr_spearman = {stu.corr_spearman(x,y)}")
        
                if return_data:
                    if c1 not in data:
                        data[c1] = dict()
                    data[c1][c2] = dict(x=x,y=y)
                ax = sml.hist2D(x,y,n_bins=bins)
                ax.set_xlabel(c1)
                ax.set_ylabel(c2)
                ax.set_title(f"{c2} vs {c1}")
                plt.show()
                #break
        #break
    if return_data:
        return data




#--- from machine_learning_tools ---
from . import pandas_ml as pdml

#--- from datasci_tools ---
from datasci_tools import matplotlib_utils as mu
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu
from datasci_tools import statistics_utils as stu

from . import seaborn_ml as sml
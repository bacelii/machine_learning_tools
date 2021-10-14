import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas_ml as pdml

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

def scatter_2D(x,y,alpha=0.5,**kwargs):
    return sns.jointplot(x,y,kind = "scatter",joint_kws={'alpha':alpha},**kwargs)

def pairplot(df,**kwargs):
    return sns.pairplot(df,**kwargs)

def hist2D(x_df,y_df,n_bins = 100,cbar=True,**kwargs):
    sns.histplot(x=x_df,#.iloc[:1000], 
           y=y_df,#.iloc[:1000],
            bins=n_bins,
             cbar=True,
                 **kwargs
           )

import seaborn_ml as sml
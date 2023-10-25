
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics

def confusion_matrix(
    y_true,
    y_pred,
    labels = None, #tells how to sort
    normalize = None, # could normalize across "true","pred",
    return_df = False,
    df = None
    ):
    if df is not None:
        y_true = df[y_true].to_list()
        y_pred = df[y_pred].to_list()
    
    return_matrix = metrics.confusion_matrix(
        y_true,
        y_pred,
        labels =labels,
        normalize = normalize,
    )
    
    if return_df:
        df = pd.DataFrame(return_matrix)
        if labels is not None:
            df.columns = labels
            df.index = labels
        return df
    else:
        return return_matrix
    
def normalize_confusion_matrix(
    cf_matrix,
    axis = 1,
    ):
    if pu.is_dataframe(cf_matrix):
        return pu.normalize_to_sum_1(cf_matrix)
    else:
        return  cf_matrix/ (np.sum(cf_matrix,axis=axis).reshape(len(cf_matrix),1))
def plot_confusion_matrix(
    cf_matrix,
    annot = True,
    annot_fontsize = 30,
    cell_fmt = ".2f",
    cmap = "Blues",
    vmin = 0,
    vmax = 1,
    
    #argmuments for axes 
    axes_font_size = 20,
    xlabel_rotation = 15,
    ylabel_rotation = 0,
    
    xlabels = None,
    ylabels = None,
    
    #colorbar 
    plot_colorbar = True,
    colobar_tick_fontsize = 25,
    
    ax = None,
    ):
#     if vmax == 1 and np.max(cf_matrix) > 1:
#         cf_matrix = normalize_confusion_matrix(cf_matrix)
    
    ax = sns.heatmap(
        cf_matrix,
        annot=annot,
        fmt = cell_fmt,
        annot_kws={
            "fontsize":annot_fontsize,
        },
        cbar = plot_colorbar,
        cmap = cmap,
        vmin=vmin, 
        vmax=vmax,
        ax = ax,
    )

    ax = mu.set_axes_font_size(
        ax,
        axes_font_size,
        x_rotation=xlabel_rotation,
        y_rotation = ylabel_rotation)
    
    if xlabels is not None or ylabels is not None:
        mu.set_axes_ticklabels(ax,xlabels,ylabels)
        
    if plot_colorbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=colobar_tick_fontsize)

    return ax
    


#--- from datasci_tools ---
from datasci_tools import matplotlib_utils as mu
from datasci_tools import pandas_utils as pu

from . import evaluation_metrics_utils as emu
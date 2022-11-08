import sklearn.metrics as metrics
import pandas as pd

def confusion_matrix(
    y_true,
    y_pred,
    labels = None, #tells how to sort
    normalize = None, # could normalize across "true","pred",
    return_df = False
    ):
    
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
    
import seaborn as sns
import matplotlib_utils as mu
import numpy as np
import pandas_utils as pu
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
        
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=colobar_tick_fontsize)

    return ax
    
import evaluation_metrics_utils as emu

from IPython.display import SVG
from IPython.display import display   
from graphviz import Source
from matplotlib.colors import ListedColormap
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def color_list_for_y(y,color_list = None):
    try:
        y = y.to_numpy()
    except:
        pass
    y_unique = np.unique(y)
    
    if color_list is None:
        color_list =  mu.generate_non_randon_named_color_list(len(y_unique))
        
    palette = {k:v for k,v in zip(y_unique,color_list)}
    return [palette[k] for k in y]
        
    
def meshgrid_for_plot(
    axes_min_default = -10,
    axes_max_default = 10,
    axes_step_default = 1,
    axes_min_max_step_dict = None,
    n_axis = None,
    return_combined_coordinates = True,
    clf = None,
    ):
    """
    Purpose: To generate a meshgrid for plotting
    that is configured as a mixutre of custom
    and default values
    
    axes_min_max_step_dict must be a dictionary mapping
    the class label or classs index to a 
    
    
    Ex: 
    vml.meshgrid_for_plot(
    axes_min_default = -20,
    axes_max_default = 10,
    axes_step_default = 1,
    #axes_min_max_step_dict = {1:[-2,2,0.5]},
    axes_min_max_step_dict = {1:dict(axis_min = -2,axis_max = 3,axis_step = 1)},
    n_axis = 2,
    clf = None,
    )
    """
    if n_axis is None:
        n_axis = clf.n_features_in_
    
    axes_default = [np.arange(axes_min_default,axes_max_default+0.01,axes_step_default)]*n_axis
    if axes_min_max_step_dict is not None:
        for axis_key,axis_value in axes_min_max_step_dict.items():
            if type(axis_key) == str:
                axis_idx = np.where(clf.classes_ == axis_key)[0][0]
            else:
                axis_idx = axis_key

            if type(axis_value) != dict:
                try:
                    a_min = axis_value[0]
                except:
                    a_min = axes_min_default
                try:
                    a_max = axis_value[1]
                except:
                    a_max = axes_max_default
                try:
                    a_step = axis_value[2]
                except:
                    a_step = axes_step_default
            else:
                a_min = axis_value.get("axis_min",axes_min_default)
                a_max = axis_value.get("axis_max",axes_max_default)
                a_step = axis_value.get("axis_step",axes_step_default)
                
            axes_default[axis_idx] = np.arange(a_min,a_max+0.01,a_step)
    
    output_grid = np.meshgrid(*axes_default)
    if return_combined_coordinates:
        grid = np.vstack([k.ravel() for k in output_grid]).T
        return grid
    else:
        return output_grid
    
def contour_map_for_2D_classifier(
    clf,
    axes_min_default = -10,
    axes_max_default = 10,
    axes_step_default = 0.01,
    axes_min_max_step_dict = None,
    axes_limit_buffers = 0,
    

    figure_width = 10,
    figure_height = 10,

    color_type = "classes", #probability (only works if the class has 2 features)

    #arguments for probability color type
    color_prob_axis = 0,
    contour_color_map = "RdBu",

    #arguments for classes color type
    map_fill_colors = None,

    # ------ arguments for plotting training points --------
    #arguments for other scatters to plot
    training_df = None,
    training_df_class_name = "class",
    training_df_feature_names = None,
    feature_names = ("feature_1","feature_2"),
    X = None,
    y = None,
    scatter_colors=["darkorange","c"],
    ):
    
    """
    Purpose: To plot the decision boundary 
    of a classifier that is only dependent on 2 feaures

    Tried extending this to classifer of more than 2 features
    but ran into confusion on how to collapse across the 
    other dimensions of the features space

    Ex: 
    %matplotlib inline
    vml.contour_map_for_2D_classifier(ctu.e_i_model)
    
    #plotting the probability
    %matplotlib inline
    vml.contour_map_for_2D_classifier(
        ctu.e_i_model,
        color_type="probability")
    """

    features_to_plot = (0,1) #could be indexes or feature name

    labels_list = getattr(clf,"classes_",[0,1])

    axes_limit_buffers = nu.convert_to_array_like(axes_limit_buffers)
    if len(axes_limit_buffers) == 1:
        axes_limit_buffers = np.concatenate([axes_limit_buffers,axes_limit_buffers])
    



    features_idx = [np.where(np.array(labels_list) == k)[0][0]
                    if type(k) == str  else k
                    for k in features_to_plot]

    #if color_type == "classes":
    if map_fill_colors is None:
        map_fill_colors = mu.generate_non_randon_named_color_list(len(labels_list))
        
    cmap_light = ListedColormap(map_fill_colors)
    


    if scatter_colors is None:
        scatter_colors = map_fill_colors


    if X is None:
        output_grid  = vml.meshgrid_for_plot(
                    axes_min_default = axes_min_default,
                    axes_max_default = axes_max_default,
                    axes_step_default = axes_step_default,
                    axes_min_max_step_dict = axes_min_max_step_dict,
            clf = clf,
            return_combined_coordinates=False

                    )
    else:
        n_mesh_grid_pts = 100
        x_min,x_max = X.iloc[:,0].min()-axes_limit_buffers[0],X.iloc[:,0].max()+axes_limit_buffers[0]
        y_min,y_max = X.iloc[:,1].min()-axes_limit_buffers[1],X.iloc[:,1].max()+axes_limit_buffers[1]
        output_grid = vml.meshgrid_for_plot(
            axes_min_max_step_dict = {0:dict(axis_min = x_min,axis_max = x_max,axis_step = (x_max-x_min)/n_mesh_grid_pts),
                                    1:dict(axis_min = y_min,axis_max = y_max,axis_step = (y_max-y_min)/n_mesh_grid_pts)},
            clf = clf,
            return_combined_coordinates=False,
        )
        
        

        

    xx,yy = output_grid[features_idx[0]],output_grid[features_idx[1]]
    grid = np.vstack([k.ravel() for k in output_grid]).T #makes into n by n_features array

    f, ax = plt.subplots(figsize=(figure_width, 
                                 figure_height))

    if color_type == "probability":
        if type(color_prob_axis) == str:
            color_prob_axis = np.where(np.array(labels_list) == color_prob_axis)[0][0]

        curr_label_for_prob = labels_list[color_prob_axis]

        probs = clf.predict_proba(grid)[:, color_prob_axis].reshape(xx.shape)

        contour = ax.contourf(xx, yy, probs, 25, cmap=contour_color_map,
                          vmin=0, vmax=1)

        ax_c = f.colorbar(contour)
        ax_c.set_label(f"$P(y = {curr_label_for_prob})$")
        ax_c.set_ticks([0, .25, .5, .75, 1])
    elif color_type == "classes":
        Z = clf.predict(grid)
        classes_map = {v:k for k,v in enumerate(labels_list)}
        Z = np.array([classes_map[k] for k in Z])
        Z = Z.reshape(xx.shape)
        contour = ax.contourf(xx, yy, Z, cmap=cmap_light)
        ax_c = f.colorbar(contour)
        ax_c.set_ticks([k for k in range(len(labels_list))])

    else:
        raise Exception(f"Unimplemented color_type = {color_type}")

    if training_df is not None or X is not None:
        if X is None:
            X,y = pdml.X_y(training_df,training_df_class_name)
        
        if training_df_feature_names is not None:
            try:
                X = X[training_df_feature_names]
            except:
                X# = X[training_df_feature_names]
        
        feature_names = list(X.columns)
            
        try:
            X = X.to_numpy()
        except:
            pass
        ax.scatter(X[:,0],
                    X[:,1],
                    c = vml.color_list_for_y(y,scatter_colors),
                    alpha = 0.5,
                    )
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        
    
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
        
    plt.show()

def plot_df_scatter_3d_classification(
    df=None,
    target_name = None,
    feature_names = None,
    
    #plotting features
    figure_width = 10,
    figure_height = 10,
    alpha = 0.5,
    axis_append = "",
    
    verbose = False,
    X=None,
    y = None,
    title = None,
    ): 
    """
    Purpose: To plot features in 3D
    
    Ex: 
    %matplotlib notebook
    sys.path.append("/machine_learning_tools/machine_learning_tools/")
    from machine_learning_tools import visualizations_ml as vml
    vml.plot_df_scatter_3d_classification(df,target_name="group_label",
                                         feature_names= [
        #"ipr_eig_xz_to_width_50",
        "center_to_width_50",
        "n_limbs",
        "ipr_eig_xz_max_95"
    ])
    """
    if target_name is None:
        target_name = "target"
    if df is None:
        if y is None:
            y = ["None"]*len(X)
        df = pdml.df_from_X_y(X,y,target_name = target_name)
    
    
    fig = plt.figure()
    fig.set_size_inches(figure_width,figure_height)
    ax = fig.add_subplot(111, projection = "3d")
    
    split_dfs = pdml.split_df_by_target(df,target_name = target_name)
    for df_curr in split_dfs:
        X,y = pdml.X_y(df_curr,target_name=target_name)
        
        curr_label = np.unique(y.to_numpy())[0]
        if verbose:
            print(f"Working on label: {curr_label}")
    
        if feature_names is None:
            feature_names = pdml.feature_names(X)

#         if verbose:
#             print(f"feature_names = {feature_names}")
            

        X_curr,Y_curr,Z_curr = [X[k].to_numpy() for i,k in enumerate(feature_names)
                               if i < 3]
        
        ax.scatter(X_curr,Y_curr,Z_curr,
                   label=curr_label,
                   alpha = alpha)
        
    
    label_function = [ax.set_xlabel,ax.set_ylabel,ax.set_zlabel]
    for lfunc,ax_title in zip(label_function,feature_names):
        lfunc(f"{ax_title} {axis_append}")
    
    
    if title is None:
        ax.set_title(" vs. ".join(np.flip(feature_names)))
    else:
        ax.set_title(title)
    mu.set_legend_outside_plot(ax)
    ax.legend()
    
    return ax

def plot_df_scatter_2d_classification(
    df=None,
    target_name = None,
    feature_names = None,
    
    #plotting features
    figure_width = 10,
    figure_height = 10,
    alpha = 0.5,
    axis_append = "",
    
    verbose = False,
    X=None,
    y = None,
    title=None,
    
    ):
    """
    Purpose: To plot features in 3D
    
    Ex: 
    %matplotlib notebook
    sys.path.append("/machine_learning_tools/machine_learning_tools/")
    from machine_learning_tools import visualizations_ml as vml
    vml.plot_df_scatter_3d_classification(df,target_name="group_label",
                                         feature_names= [
        #"ipr_eig_xz_to_width_50",
        "center_to_width_50",
        "n_limbs",
        "ipr_eig_xz_max_95"
    ])
    
    
    Ex: 
    ax = vml.plot_df_scatter_2d_classification(
        X=X_trans[y!= "Unknown"],
        y = y[y != "Unknown"],
    )

    from datasci_tools import matplotlib_utils as mu
    mu.set_legend_outside_plot(ax)
    
    """
    if target_name is None:
        target_name = "target"
    if df is None:
        df = pdml.df_from_X_y(X,y,target_name = target_name)
    
    
    fig,ax= plt.subplots(1,1)
    fig.set_size_inches(figure_width,figure_height)
    
    split_dfs = pdml.split_df_by_target(df,target_name = target_name)
    for df_curr in split_dfs:
        X,y = pdml.X_y(df_curr,target_name=target_name)
        
        curr_label = np.unique(y.to_numpy())[0]
        if verbose:
            print(f"Working on label: {curr_label}")
    
        if feature_names is None:
            feature_names = pdml.feature_names(X)

#         if verbose:
#             print(f"feature_names = {feature_names}")
            

        X_curr,Y_curr = [X[k].to_numpy() for i,k in enumerate(feature_names)
                        if i < 2]
        
        ax.scatter(X_curr,Y_curr,
                   label=curr_label,
                   alpha = alpha)
        
    
    label_function = [ax.set_xlabel,ax.set_ylabel]
    for lfunc,ax_title in zip(label_function,feature_names):
        lfunc(f"{ax_title} {axis_append}")
    
    if title is None:
        ax.set_title(" vs. ".join(np.flip(feature_names)))
    else:
        ax.set_title(title)
    mu.set_legend_outside_plot(ax)
    ax.legend()
    
    return ax

'''def plot_df_scatter_classification_old(
    df=None,
    target_name = None,
    feature_names = None,
    
    #plotting features
    figure_width = 10,
    figure_height = 10,
    alpha = 0.5,
    axis_append = "",
    
    verbose = False,
    X=None,
    y = None,
    title = None,
    **kwargs
    ):
    """
    Ex: 
    
    ax = vml.plot_df_scatter_classification(
    X=X_trans[y!= "Unknown"][:,:2],
    y = y[y != "Unknown"],
    )

    from datasci_tools import matplotlib_utils as mu
    ax = mu.set_legend_outside_plot(ax)
    """
    
    if target_name is None:
        target_name = "target"
    if df is None:
        if y is None:
            y = ["None"]*len(X)
        df = pdml.df_from_X_y(X,y,target_name = target_name)
    
    if len(df.columns) <= 3:
        plot_func = plot_df_scatter_2d_classification
    else:
        plot_func = plot_df_scatter_3d_classification
        
    if verbose:
        print(f"Using plotting function {plot_func}")
    return plot_func(
        df=df,
        target_name = target_name,
        feature_names = feature_names,

        #plotting features
        figure_width = figure_width,
        figure_height = figure_height,
        alpha = alpha,
        axis_append = axis_append,

        verbose = verbose,
        X=X,
        y = y,
        title = title,
        **kwargs
    )'''
    
def plot_df_scatter_classification(
    df=None,
    target_name = None,
    feature_names = None,
    ndim = 3,
    
    #plotting features
    figure_width = 14,
    figure_height = 14,
    alpha = 0.5,
    axis_append = "",
    
    verbose = False,
    X=None,
    y = None,
    title=None,
    target_to_color = None,
    default_color = "yellow",
    
    #for the legend:
    plot_legend = True,
    scale_down_legend = 0.75,
    bbox_to_anchor=(1.02,0.5),
    
    ax = None,

    
    #--- text to plot ----
    
    #--  a dictionary mapping the name o a coordinate to plot at
    text_to_plot_dict=None,
    
    # --- whether to compute the dictionary of text to plot from labels
    use_labels_as_text_to_plot = False,
    
    #labels to plot for each of he datapoints
    text_to_plot_individual = None,
    
    replace_None_with_str_None = False,
    
    #for text parameters:
    **kwargs
    ):
    """
    Purpose: To plot features in 3D
    
    Ex: 
    %matplotlib notebook
    sys.path.append("/machine_learning_tools/machine_learning_tools/")
    from machine_learning_tools import visualizations_ml as vml
    vml.plot_df_scatter_3d_classification(df,target_name="group_label",
                                         feature_names= [
        #"ipr_eig_xz_to_width_50",
        "center_to_width_50",
        "n_limbs",
        "ipr_eig_xz_max_95"
    ])
    """
    if target_name is None:
        target_name = "target"
    if df is None:
        if y is None:
            y = ["None"]*len(X)
        df = pdml.df_from_X_y(X,y,target_name = target_name)
    
        
    if len(df.columns) <= 3:
        ndim = 2
    
    if ax is None:
        if ndim == 3:
            fig = plt.figure()
            fig.set_size_inches(figure_width,figure_height)
            ax = fig.add_subplot(111, projection = "3d")
        elif ndim == 2:
            fig,ax= plt.subplots(1,1)
            fig.set_size_inches(figure_width,figure_height)
        else:
            raise Exception("")
        
    if replace_None_with_str_None:
        df = pu.replace_None_with_default(df,"None")
    
    split_dfs = pdml.split_df_by_target(df,target_name = target_name)
    for df_curr in split_dfs:
        X,y = pdml.X_y(df_curr,target_name=target_name)
        
        curr_label = np.unique(y.to_numpy())[0]
        if verbose:
            print(f"Working on label: {curr_label}")
    
        if feature_names is None:
            feature_names = pdml.feature_names(X)
            
        if target_to_color is not None:
            c = target_to_color.get(curr_label,default_color) 
            #c = mu.color_to_rgb(c)
        else:
            c = default_color

#         if verbose:
#             print(f"feature_names = {feature_names}")
            

        curr_data = [X[k].to_numpy() for i,k in enumerate(feature_names)
                               if i < ndim] 
        
        ax.scatter(*curr_data,
                   label=curr_label,
                   c = c,
                   alpha = alpha
        )
        
    if ndim == 3:
        label_function = [ax.set_xlabel,ax.set_ylabel,ax.set_zlabel]
    else:
        label_function = [ax.set_xlabel,ax.set_ylabel]
    for lfunc,ax_title in zip(label_function,feature_names):
        lfunc(f"{ax_title} {axis_append}")
        
        
    # Doing the plotting of the labels
    
    
    if text_to_plot_dict is not None or use_labels_as_text_to_plot: 
        X,y = pdml.X_y(df,target_name=target_name)
        mu.text_overlay(
            ax,
            text_to_plot_dict=text_to_plot_dict,
            X=X,
            y = y,
            **kwargs
        )

    
    
    if title is None:
        ax.set_title(" vs. ".join(np.flip(feature_names)))
    else:
        ax.set_title(title)
    
    if plot_legend:
        ax.legend()
        mu.set_legend_outside_plot(
            ax,
            #scale_down=scale_down_legend,
            bbox_to_anchor = bbox_to_anchor,
            #loc = "center left"
        )
    
    
    
    return ax
    


    
    
#Plotting function
def plot_svm_kernels(clf, X, y, X_test=None,title = None):
#     import warnings
#     import logging,sys
#     warnings.filterwarnings('ignore')
#     logging.disable(sys.maxsize)
    
    #Plot
    plt.figure()
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    if X_test is not None:
        plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                    zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    if title is not None:
        plt.title(title)
    else:
        plt.title(str(clf.kernel))
    plt.show()
    
def plot_feature_importance(clf,
                            feature_names = None,
                           sort_features=True,
                           n_features_to_plot=20,
                           title="Feature Importance",
                            importances=None,
                            std=None,
                           **kwargs):
    """
    Purpose: Will plot the feature importance of a classifier
    """
    if importances is None or std is None:
        importances,std = sklm.feature_importances(clf,
                                                   return_std=True,
                                                   **kwargs)
        
    if feature_names is None:
        feature_names= np.array([f"feature_{i}" for i in range(len(importances))])
    
    if sort_features:
        sort_idx = np.flip(np.argsort(importances))
        importances = importances[sort_idx]
        feature_names = feature_names[sort_idx]
    
    df = pd.DataFrame(dict(importances=importances,index=feature_names,std=std))
    
    #doing the filtering of the features
    if n_features_to_plot is not None and len(df)>n_features_to_plot:
        df = df.iloc[:n_features_to_plot,:]

    #doing the plotting
    fig,ax = plt.subplots()
    df.plot.bar(x="index",y="importances",yerr="std",ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    
    
    


def plot_decision_tree(clf,
                      feature_names,
                      class_names=None,
                      max_depth=None):
    """
    Purpose: Will show the 
    
    
    """
    if class_names is None:
        class_names = list(clf.classes_)
    
    graph = Source(export_graphviz(clf,
                        max_depth=max_depth
      , out_file=None
      , feature_names=feature_names
      , class_names=class_names
      , filled = True
      , precision=6))

    print(f"clf.classes_ = {clf.classes_}")
    display(SVG(graph.pipe(format='svg')))
    
    
def plot_dim_red_analysis(
    X,
    method,
    y = None,
    n_components = [2,3],
    alpha = 0.5,
    color_mapppings = None,
    plot_kwargs = None,#dict(),
    verbose=False,
    **kwargs
    ):
    
    if plot_kwargs is None:
        plot_kwargs = dict()
    
    n_components = nu.convert_to_array_like(n_components)
    color_mapppings = nu.convert_to_array_like(color_mapppings)
    
    for ndim in n_components:
        X_trans = dru.dimensionality_reduction_by_method(
        method=method,
        X = X,
        n_components =ndim,
        **kwargs
        )
        
        for cm in color_mapppings:
            vml.plot_df_scatter_classification(
                X = X_trans,
                y = y,
                target_to_color = cm,
                ndim = ndim,
                alpha = alpha,
                title=method,
                **plot_kwargs
                
            )
            plt.show()
    
    
def plot_binary_classifier_map(
    clf,
    X = None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    buffer = 0.01,
    class_idx_to_plot = 0,
    #plotting
    figure_width = 10,
    figure_height = 10,
    axes_fontsize = 25,

    class_0_color = None,#mu.colorblind_blue,
    class_1_color = None,#mu.colorblind_orange,
    mid_color = "white",
    alpha = 0.5,
    plot_colorbar = True,
    colorbar_label = None,
    colorbar_labelpad = 30,
    colorbar_label_fontsize = 20,
    colorbar_tick_fontsize = 25,
    
    ax = None,
    **kwargs
    ):
    """

    Purpose: To plot the prediction
    map of a binary classifier

    Arguments: 
    1) Model
    2) Define the input space want to test over
    (xmin,xmax)

    Pseudocode: 
    1) Create a meshgrid of the input space
    2) Send the prediction


    ======Example:=======
    from machine_learning_tools import visualizations_ml as vml

    figure_width = 10
    figure_height = 10
    fig,ax = plt.subplots(1,1,figsize=(figure_width,figure_height))

    X = df_plot[trans_cols].to_numpy().astype("float")

    vml.plot_binary_classifier_map(
        clf = ctu.e_i_model,
        X = X,
        xmin = 0,
        xmax = 4.5,
        ymin=-0.1,
        ymax = 1.2,
        alpha = 0.5,
        colorbar_label = "Excitatory Probability",
        ax = ax,
    )
    """
    if class_0_color is None:
        class_0_color = mu.colorblind_blue
    if class_1_color is None:
        class_1_color = mu.colorblind_orange
    
    
    
    
    if xmin is None:
        xmin = X[:, 0].min() - buffer

    if xmax is None:
        xmax = X[:, 0].max() + buffer

    if ymin is None:
        ymin = X[:, 1].min() - buffer

    if ymax is None:
        ymax = X[:, 1].max() + buffer


    XX,YY = nu.grid_array(xmin,xmax,ymin,ymax,n_intervales = 1000)
    X = np.vstack([XX.ravel(),YY.ravel()]).T
    Z = clf.predict_proba(X)[:,class_idx_to_plot]

    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(figure_width,figure_height))

    colors = np.array([class_0_color,class_1_color])[[1-class_idx_to_plot,class_idx_to_plot]]
    cmap = mu.cmap_from_color_a_to_b(
        colors[0],
        colors[1],
        mid_color,
    )

    #clev = np.arange(-0.5,0.5,.0001)
    if colorbar_label is None:
        colorbar_label = f"Class {class_idx_to_plot} Probability"
        
    mu.plot_heatmap(
        XX,YY,Z,
        cmap = cmap,
        ax = ax,
        alpha = alpha,

        # colorbar
        plot_colorbar = plot_colorbar,
        colorbar_label = colorbar_label,
        colorbar_labelpad = colorbar_labelpad,
        colorbar_label_fontsize = colorbar_label_fontsize,
        colorbar_tick_fontsize=colorbar_tick_fontsize,
        **kwargs
    )
    
    
    
    return ax





#--- from machine_learning_tools ---
from . import dimensionality_reduction_ml as dru
from . import evaluation_metrics_utils as emu
from . import matplotlib_ml as mu
from . import numpy_ml as nu
from . import pandas_ml as pdml
from . import sklearn_models as sklm

plot_confusion_matrix = emu.plot_confusion_matrix
    

#--- from datasci_tools ---
from datasci_tools import matplotlib_utils as mu
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu

from . import visualizations_ml as vml
import pandas_ml as pdml
import numpy as np
import matplotlib_ml as mu
import matplotlib.pyplot as plt


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
    
from matplotlib.colors import ListedColormap
import pandas_ml as pdml
def contour_map_for_2D_classifier(
    clf,
    axes_min_default = -10,
    axes_max_default = 10,
    axes_step_default = 0.01,
    axes_min_max_step_dict = None,

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

    labels_list = clf.classes_



    features_idx = [np.where(np.array(labels_list) == k)[0][0]
                    if type(k) == str  else k
                    for k in features_to_plot]

    #if color_type == "classes":
    if map_fill_colors is None:
        map_fill_colors = mu.generate_non_randon_named_color_list(len(labels_list))
        
    cmap_light = ListedColormap(map_fill_colors)
    


    if scatter_colors is None:
        scatter_colors = map_fill_colors



    output_grid  = vml.meshgrid_for_plot(
                axes_min_default = axes_min_default,
                axes_max_default = axes_max_default,
                axes_step_default = axes_step_default,
                axes_min_max_step_dict = axes_min_max_step_dict,
        clf = clf,
        return_combined_coordinates=False

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
                X = X[training_df_feature_names]
            
        try:
            X = X.to_numpy()
        except:
            pass
        ax.scatter(X[:,0],
                    X[:,1],
                    c = vml.color_list_for_y(y,scatter_colors),
                    alpha = 0.5,
                    )
        ax.set_xlim(np.min(X[:,0])-0.5,np.max(X[:,0])+0.5,)
        ax.set_ylim(np.min(X[:,1])-0.5,np.max(X[:,1])+0.5,)
        
    ax.set_xlabel(training_df_feature_names[0])
    ax.set_ylabel(training_df_feature_names[1])
        
    plt.show()

def plot_df_scatter_3d_classification(
    df,
    target_name = None,
    feature_names = None,
    
    #plotting features
    figure_width = 10,
    figure_height = 10,
    alpha = 0.5,
    axis_append = "",
    
    verbose = False,
    ):
    """
    Purpose: To plot features in 3D
    
    Ex: 
    %matplotlib notebook
    sys.path.append("/machine_learning_tools/machine_learning_tools/")
    import visualizations_ml as vml
    vml.plot_df_scatter_3d_classification(df,target_name="group_label",
                                         feature_names= [
        #"ipr_eig_xz_to_width_50",
        "center_to_width_50",
        "n_limbs",
        "ipr_eig_xz_max_95"
    ])
    """
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
            

        X_curr,Y_curr,Z_curr = [X[k].to_numpy() for k in feature_names]
        
        ax.scatter(X_curr,Y_curr,Z_curr,
                   label=curr_label,
                   alpha = alpha)
        
    
    label_function = [ax.set_xlabel,ax.set_ylabel,ax.set_zlabel]
    for lfunc,ax_title in zip(label_function,feature_names):
        lfunc(f"{ax_title} {axis_append}")
    
    
    ax.set_title(" vs. ".join(np.flip(feature_names)))
    mu.set_legend_outside_plot(ax)
    ax.legend()
    
    
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
    
    
import visualizations_ml as vml
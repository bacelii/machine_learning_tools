# ---------- Decision Trees ------------ #
import pandas as pd
from sklearn.tree import DecisionTreeClassifier #
from sklearn.model_selection import train_test_split #helps with splitting data
from sklearn import metrics #will help with calculating accuracy

def decision_tree_sklearn(df,
                          target_column,
                         feature_columns=None,
                          perform_testing=False,
                          test_size = 0,
                          
                          # parameters for the decision tree
                            criterion = "entropy", # entropy for infromation gain, gini fro gini index
                            splitter = "best", #For the splitt strategy,also can be "random"
                            max_depth = None,
                            max_features = None,
                            min_samples_split= 0.1,
                            min_samples_leaf = 0.02,
                          
                         ):
    
    """
    Purpose: To train a decision tree
    based on a dataframe with the features and the classifications
    
    Parameters:
    max_depth = If None then the depth is chosen so all leaves contin less than min_samples_split
                            The higher the depth th emore overfitting


    
    """
    
    if feature_columns is None:
        feature_columns = list(df.columns)
        feature_columns.remove(target_column)
        
    
    col_names = feature_columns
    
    #1) dividing out the data into features and classifications
    X = df[col_names]
    y = df[[target_column]]
    
    
    #2) Dividing the data into test and training set
    test_size = 0
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=1)
    else:
        X_train = X_test = X
        y_train = y_test = y

        
        
    
    #3) Train the classifier
    clf = DecisionTreeClassifier(criterion=criterion,
                                splitter=splitter,
                                max_depth = max_depth,
                                max_features=max_features,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf
                                )

    # training the model
    clf = clf.fit(X_train,y_train)

    
    if perform_testing and test_size > 0:
        #testing the trained model
        y_pred = clf.predict(X_test)

        #measure the accuracy
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    return clf

# How to visualize tree 
from IPython.display import SVG
from graphviz import Source
from IPython.display import display   

from sklearn.tree import export_graphviz

def plot_decision_tree(clf,
                      feature_names,
                      class_names=None):
    """
    Purpose: Will show the 
    
    
    """
    if class_names is None:
        class_names = list(clf.classes_)
    
    graph = Source(export_graphviz(clf
      , out_file=None
      , feature_names=feature_names
      , class_names=class_names
      , filled = True
      , precision=6))

    print(f"clf.classes_ = {clf.classes_}")
    display(SVG(graph.pipe(format='svg')))
    
    
import numpy as np
def print_tree_structure_description(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    
    
#     if feature_names is None:
#         feature = clf.tree_.feature
#     else:
#         feature = feature_names
        
        
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has {n} nodes and has "
          "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i))
        else:
            print("{space}node={node} is a split node: "
                  "go to node {left} if X[:, {feature}] <= {threshold} "
                  "else to node {right}.".format(
                      space=node_depth[i] * "\t",
                      node=i,
                      left=children_left[i],
                      feature=feature[i],
                      threshold=threshold[i],
                      right=children_right[i]))

            
def decision_tree_analysis(df,
                          target_column,
                          max_depth=None,
                          max_features=None,):
    """
    Purpose: To perform decision tree analysis and plot it
    """
    feature_columns = list(df.columns)
    feature_columns.remove(target_column)

    clf = mlu.decision_tree_sklearn(df=df,
                              target_column=target_column,
                             feature_columns=None,
                              perform_testing=False,
                              test_size = 0,

                              # parameters for the decision tree
                                criterion = "entropy", # entropy for infromation gain, gini fro gini index
                                splitter = "best", #For the splitt strategy,also can be "random"
                                max_depth = max_depth,
                                max_features = max_features,
                                min_samples_split= 0.1,
                                min_samples_leaf = 0.02,

                             )
    mlu.plot_decision_tree(clf,
                       feature_names=feature_columns,
                      )
    
    
from sklearn import preprocessing
def encode_labels_as_ints(labels):
    """
    Purpose: Will convert a list of labels into 
    an array encoded 0,1,2.... and return the unique
    labels used
    
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
    
    Ex: 
    import machine_learning_utils as mlu
    mlu.encode_labels_as_ints(["hi","hello","yo",'yo'])
    """
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    
    return le.transform(labels),le.classes_

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from datasci_tools import matplotlib_utils as mu
def kNN_classifier(
    X, #input features
    y, #output label (in 0,1,2...)
    n_neighbors,
    labels_list = None,
    weights = "distance", #other is uniform
    verbose = False,
    
    #for plotting features
    plot_map = False,
    add_labels = True,
    
    **kwargs
    
    ):
    """
    Purpose: Will create a kNN classifier and 
    preferrencially plot the decision map
    
    Ex: Running it on the iris data
    
    from sklearn import neighbors, datasets
    import machine_learning_utils as mlu

    n_neighbors = 15

    # import some data to play with
    iris = datasets.load_iris()

    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    mlu.kNN_classifier(X,y,
                       n_neighbors=n_neighbors,
                      labels_list = iris.target_names,
                       plot_map = True,
                       feature_1_name=iris.feature_names[0],
                       feature_2_name=iris.feature_names[1],
                      )
    
    """
    
    
    y = np.array(y)
    
    if labels_list is None:
        labels_list = np.array([f"type_{k}" for k in np.unique(y)])
    
    if not "int" in str(y.dtype):
        y, labels_list = mlu.encode_labels_as_ints(y)
        
    if verbose:
        for i,lab in enumerate(labels_list):
            print(f"{i}:{lab}")
        
        
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    
    if verbose:
        print(f"X.shape = {X.shape}, y.shape = {y.shape}")
        
    clf.fit(X, y)
    
    if add_labels:
        clf.labels = labels_list

    if plot_map:
        mlu.plot_classifier_map(clf,**kwargs)
        
    return clf


from datasci_tools import numpy_utils as nu    
def plot_classifier_map(
    clf,
    data_to_plot=None,
    data_to_plot_color = 'red',
    data_to_plot_size = 50,
    figsize = (8, 6),
    map_fill_colors = None,#['orange', 'cyan', 'cornflowerblue'],
    scatter_colors = None,#['darkorange', 'c', 'darkblue'],
    h = .02,  # step size in the mesh
    feature_1_idx = 0,
    feature_2_idx = 1,
    feature_1_name = "feature_1_name",
    feature_2_name = "feature_2_name",
    x_min = None,
    x_max = None,
    y_min = None,
    y_max = None,
    verbose = False,
    plot_training_points = False,
    **kwargs):
    """
    Purpose: To plot
    """
    
    print(f"data_to_plot = {data_to_plot}")
    X = clf._fit_X
    y = clf._y
    try:
        labels_list = clf.labels
    except:
        labels_list = clf.classes_

    # Create color maps
    if map_fill_colors is None:
        map_fill_colors = mu.generate_non_randon_named_color_list(len(labels_list))
    if scatter_colors is None:
        scatter_colors = map_fill_colors
    
    cmap_light = ListedColormap(map_fill_colors)
    cmap_bold = scatter_colors


    if x_min is None:
        x_min = X[:, feature_1_idx].min() - 1

    if x_max is None:
        x_max = X[:, feature_1_idx].max() + 1

    if y_min is None:
        y_min = X[:, feature_2_idx].min() - 1

    if y_max is None:
        y_max = X[:, feature_2_idx].max() + 1

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # collecting the mesh points into datapoints that are used for prediction
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=figsize)
    if Z.dtype == np.dtype("O"):
        labels_list,Z = np.unique(Z,return_inverse=True)
        Z = Z.reshape(yy.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light)


    # Plot also the training points
    if plot_training_points:
        try:
            palette = {k:v for k,v in zip(labels_list,cmap_bold)}
            ax2 = sns.scatterplot(x=X[:, feature_1_idx], 
                            y=X[:, feature_2_idx],
                            #color=np.array(cmap_bold)[y],
                            hue=labels_list[y],
                            #palette=cmap_bold, 
                            palette = palette,
                            alpha=1.0, 
                            edgecolor="black")
        except:
            palette = {k:v for k,v in zip(labels_list,cmap_bold)}
            ax2 = sns.scatterplot(x=X[:, feature_1_idx], 
                            y=X[:, feature_2_idx],
                            #color=np.array(cmap_bold)[y],
                            hue=y,
                            #palette=cmap_bold, 
                            palette = palette,
                            alpha=1.0, 
                            edgecolor="black")
    else: 
        ax2 = plt.gca()
        
    
    if data_to_plot is not None:
        data_to_plot = np.array(data_to_plot).reshape(-1,2)
        data_to_plot_color= nu.convert_to_array_like(data_to_plot_color)
        if len(data_to_plot_color) == 1:
            data_to_plot_color = list(data_to_plot_color)*len(data_to_plot)
        #print(f"data_to_plot_color = {data_to_plot_color}")
        ax2.scatter(data_to_plot[:,0],
                   data_to_plot[:,1],
                   c = data_to_plot_color,
                   marker = "x",
                   s = data_to_plot_size,
                   )
        print(f"Attempting to plot data")
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"{len(labels_list)}-Class classification")
    plt.xlabel(feature_1_name)
    plt.ylabel(feature_2_name)
    
    if verbose:
        print(f"Parameters for the model = {clf.get_params()}")

    plt.show()

from datasci_tools import system_utils as su
def export_model(model,path):
    su.compressed_pickle(model,path)

def load_model(path):
    return su.decompress_pickle(path)

def predict_class_single_datapoint(clf,
                                  data,
                                  verbose = False,
                                  return_probability = False):
    """
    Purpose: To predict the class of a single datapoint
    
    Ex: 
    data = [1,1]
    mlu.predict_class_single_datapoint(clf,data,verbose = True)

    """
    data = np.array(data).reshape(1,-1)
    pred_class = clf.predict(data)[0]
    pred_prob = np.max(clf.predict_proba(data)[0])
    if verbose:
        print(f"Class = {pred_class} for data = {data} with prediction probability {pred_prob}")
        
    if return_probability:
        return pred_class,pred_prob
    else:
        return pred_class
    
from . import machine_learning_utils as mlu
import numpy as np
import matplotlib.pyplot as plt


def updated_cluster_centers(data,labels,n_clusters):
    """
    Calculate the new cluster centers for k-means
    by averaging the data points of those
    assigned to each cluster
    
    """
    current_cluster_centers = []
    for ki in range(n_clusters):
        cl_center=np.mean(data[np.where(labels==ki)[0]],axis=0)
        current_cluster_centers.append(cl_center)
    return np.array(current_cluster_centers)
        
def calculate_k_means_loss(data,labels,cluster_centers):
    """
    Purpose: Will calculate the k means loss that depends on:
    1) cluster centers
    2) current labels of data
    
    Pseudocode: 
    For each datapoint:
        a) Calculate the squared euclidean distance between datapoint and center of cluster
           it is assigned to
    Sum up all of the squared distances
    """
    #total_data_point_losses = [np.linalg.norm(xi-cluster_centers[ki])**2 for xi,ki in zip(data,labels)]
    total_data_point_losses = np.linalg.norm(cluster_centers[labels]-data)**2
    return np.sum(total_data_point_losses)
    
def reassign_data_to_clusters(data,cluster_centers):
    data_dist_to_centers = np.array([np.linalg.norm(cluster_centers-xi,axis=1)**2 for xi in data])
    min_dist_label = np.argmin(data_dist_to_centers,axis=1)
    return min_dist_label
    


def k_mean_clustering(data,
                     n_clusters=3,
                     max_iterations = 1000,
                    return_snapshots = True,
                     verbose = True):

    """ 
    Purpose: Will take in the input data
    and the number of expected clusters and 
    run the K-Means algorithm to cluster the data
    into k-clusters

    Arguments: 
    - data (np.array): the data points in R40 to be clusters
    - n_clusters: Number of expected clusters
    - max_iterations: upper bound on number of iterations
    - return_snapshots: return the label assignments, cluster centers and loss value
                        for every iteration

    Returns: 
    - final_data_labels: what cluster each datapoint was assigned to at end
    - final_cluster_centers: cluster centers on last iteration
    - final_loss: steady-state value of loss function at end

    * snapshots of labels,centers and loss if requested

    Pseudocode: 
    1) Randomly assign labels to data
    2) Calculate the cluster centers from random labels
    3) Calculate loss value 
    4) Begin iteration loop: 
        a. Reassign data labels to the closest cluster center
        b. Recalculate the cluster center
        c. Calculate the k-means loss
        d. If the k-means loss did not change from previous value
            OR max_iterations is reached

            break out of loop
    5) return final values (and snapshots if requested)
    """


    #0) Containers to hold labels, cluster centers and loss value
    labels_history = []
    cluster_center_history = []
    loss_history = []


    #1) Randomly assign labels to data
    current_labels = np.random.randint(0,high=n_clusters,size=data.shape[0])

    #2) Calculate the cluster centers from random labels
    current_cluster_centers = updated_cluster_centers(data,
                                                      labels=current_labels,
                                                      n_clusters=n_clusters)

    #3) Calculate loss value
    current_loss = calculate_k_means_loss(data,
                                         labels=current_labels,
                                         cluster_centers=current_cluster_centers)
    if verbose:
        print(f"current_loss = {current_loss}")


    #save the current status in the history:
    labels_history.append(current_labels)
    cluster_center_history.append(current_cluster_centers)
    loss_history.append(current_loss)

    #4) Begin iteration loop: 
    break_from_loss_steady_state=False
    for i in range(0,max_iterations):
        #a. Reassign data labels to the closest cluster center
        current_labels = reassign_data_to_clusters(data,
                                  cluster_centers=current_cluster_centers)
        #b. Recalculate the cluster center
        current_cluster_centers = updated_cluster_centers(data,
                                                      labels=current_labels,
                                                      n_clusters=n_clusters)
        #c. Calculate the k-means loss
        current_loss = calculate_k_means_loss(data,
                                         labels=current_labels,
                                         cluster_centers=current_cluster_centers)
        #d. If the k-means loss did not change from previous value
        if current_loss < loss_history[-1]:
            labels_history.append(current_labels)
            cluster_center_history.append(current_cluster_centers)
            loss_history.append(current_loss)
        elif current_loss == loss_history[-1]:
            if verbose:
                print(f"Breaking out of K-Means loop on iteration {i} because steady state loss {current_loss}")
            break_from_loss_steady_state=True
            break
        else:
            raise Exception(f"The loss grew after iteration {i} from {loss_history[-1]} to {current_loss}")

    return_values = [current_labels,current_cluster_centers,current_loss]

    if return_snapshots:
        return_values.append(labels_history)
        return_values.append(cluster_center_history)
        return_values.append(loss_history)
        
    return return_values

def plot_voltage_vs_time(n_clusters,
                        final_cluster_centers,
                        final_labels,
                        data):
    """
    Purpose: 
    For each cluster:
        1) Plot the cluster center as a waveform
        2) All the waveform snippets assigned to the cluster

    """

    fig,axes = plt.subplots(n_clusters,2)
    fig.set_size_inches(20,20)

    for i in range(n_clusters):
        ax1 = axes[i][0]
        ax1.plot(final_cluster_centers[i],c="r",label="cluster_center")
        ax1.set_title(f"Cluster {i} Waveform")
        ax1.set_xlabel("Time Samples")
        ax1.set_ylabel(r"Voltage ($ \mu V $)")


        cluster_data = np.where(final_labels==i)[0]

        y_range = [np.min(data[cluster_data]),np.max(data[cluster_data])]

        ax2 = axes[i][1]
        for d in cluster_data:
            ax2.plot(data[d],c="black",alpha=0.3)

        ax2.set_title(f"Waveforms Assigned to Cluster {i}")
        ax2.set_xlabel("Time Samples")
        ax2.set_ylabel(r"Voltage ($ \mu V$)")

        ax1.set_ylim(y_range)
        ax2.set_ylim(y_range)
        
        
def plot_loss_function_history(loss_history,title="K-Means Loss vs. Iterations",
                              n_clusters=None):
    fig,ax = plt.subplots(1,1)
    ax.plot(loss_history)
    if not n_clusters is None:
        title += f" for n_clusters={n_clusters}"
    ax.set_title(title)
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Value of Loss Function: \n (Sum of Squared Euclidean Distances\n to Assigned Cluster Center)")
    
    
# ----------------------- Problem 4: For GMM --------------------

from sklearn import mixture
import pandas as pd
import seaborn as sns

def plot_4D_GMM_clusters(X_train,
                        X_test=None,
                        K=10,
                        covariance_type = 'full'):
    """
    To graph the 4D clustering of the peaks of the AP
    
    
    """
    if X_test is None:
        print(f"Plotting the clustered labels of the training data for clusters = {K}")
        X_test = X_train
    else:
        print(f"Plotting the clustered labels of the testing data for clusters = {K}")
        

    #1) Training the GMM
    gmix = mixture.GaussianMixture(n_components=K, covariance_type=covariance_type)
    gmix.fit(X_train)
    #2) Use the train parameters to make predictions for labels
    y_train_pred = gmix.predict(X_test)

    #3) Graph the predicted labels

    PP = pd.DataFrame(np.array(X_test))
    PP["cluster"] = y_train_pred 

    g = sns.PairGrid(PP,hue="cluster")
    g = g.map_lower(plt.hexbin,gridsize=50, mincnt=1, cmap='seismic',bins='log')
    for i, j in zip(*np.triu_indices_from(g.axes, 0)):
        g.axes[i, j].set_visible(False)

    g.fig.suptitle(f'Pairwise amplitudes of channels at index of peak value for average channel recording\n For number of clusters = {K} ',
                   fontsize=16,
                  y = 0.8)


def compute_average_log_likelihood_per_K(
    peaks,
    N = 5000,
    n_iterations = 10,
    K_list = np.arange(8,21),
    return_train=True,
    covariance_type='full',
    ):
    
    iter_average_log_like_test = []
    iter_average_log_like_train = []

    for i in range(n_iterations):
        print(f"--- Working on iteration {i} ---")
        average_log_like_list_test = []
        average_log_like_list_train = []
        for K in K_list:
            gmix = mixture.GaussianMixture(n_components=K, covariance_type=covariance_type)
            X_train = peaks[:N]
            gmix.fit(X_train)
            average_log_likelihood_train = gmix.score(X_train)
            average_log_like_list_train.append(average_log_likelihood_train)

            X_test = peaks[N:2*N]
            #y_test_pred = gmix.predict(X_test)

            average_log_likelihood_test = gmix.score(X_test)
            average_log_like_list_test.append(average_log_likelihood_test)

        average_log_like_list_test = np.array(average_log_like_list_test)
        average_log_like_list_train = np.array(average_log_like_list_train)


        iter_average_log_like_test.append(average_log_like_list_test)
        iter_average_log_like_train.append(average_log_like_list_train)


    iter_average_log_like_test = np.array(iter_average_log_like_test)
    iter_average_log_like_train = np.array(iter_average_log_like_train)

    iter_average_log_like_test_av = np.mean(iter_average_log_like_test,axis=0)
    iter_average_log_like_train_av = np.mean(iter_average_log_like_train,axis=0)
    
    if return_train:
        return iter_average_log_like_test_av,iter_average_log_like_train_av
    else:
        iter_average_log_like_test_av
        
        
        
# ============== Hierarchical Clustering ========================
"""
For hierarchical clusters: 

model.children_ : describes the binary tree of the clustering

"""


from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
def dendrogram_leaves_ordered(model,**kwargs):
    """
    Gets the order of the leaves in the dendrogram
    
    Applictation: For bi-clustering
    """
    dend = clu.dendrogram_HC(model,**kwargs)
    return dend["leaves"]
def dendrogram_HC(model,p = 10000,
                    no_plot = True,
                    **kwargs):
    """
    Purpose: to create a dendrogram
    and plot it for a hierarchical clustering model
    
    Ex: 
    p = 1000
    # plot the top three levels of the dendrogram
    curr_dendrogram = clu.dendrogram_HC(model,no_plot=False)

    
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    if no_plot:
        return dendrogram(linkage_matrix, p = p,
                          no_plot=no_plot,
                          **kwargs)
    else:
        plt.figure(figsize=[10,10])
        plt.title("Hierarchical Clustering Dendrogram")
        return dendrogram(linkage_matrix, p = p,
                          no_plot=no_plot,
                          **kwargs)
    
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()
        

import networkx as nx
def dendrogram_graph_from_model(model):
    """
    Purpose: will return the dendrogram as a grpah object
    so you can navigate it
    """
    edges = np.vstack([np.array([[i+model.n_leaves_,k1],
     [i+model.n_leaves_,k2]]) for i,(k1,k2) in enumerate(model.children_)])
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    return G
    
    
import networkx_utils as xu

def closest_k_nodes_on_dendrogram(
    node,
    k,
    G = None,
    model=None,
    verbose = False):
    """
    Purpose: Want to find the first k nodes that
    are close to a node through a dendrogram
    """

    if G is None:
        G = clu.dendrogram_graph_from_model(model)

    return xu.closest_k_leaf_neighbors_in_binary_tree(
        G,
        node=node,
        k = k,
        verbose = verbose
    )


from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
import clustering_ml as clu
def closet_k_neighbors_from_nierarchical_clustering(
    X,
    node_name,
    row_names,
    k,
    n_components = 3,
    verbose = False,
    ):

    text_ids = np.array(row_names)

    model = AgglomerativeClustering(
        distance_threshold=0, 
        n_clusters=None,
        linkage="ward")

    model = model.fit(X)
    
    G = clu.dendrogram_graph_from_model(model)

    graph_node = np.where(text_ids == node_name)[0][0]

    closest_neighbors = clu.closest_k_nodes_on_dendrogram(
        node = graph_node,
        k = k,
        G = G,
        verbose = verbose
    )


    return [str(k) for k in text_ids[closest_neighbors]]
    
        
from sklearn.cluster import AgglomerativeClustering


# --------------- metrics for evaluating clusters -------------
from sklearn import metrics
def purity_score(labels_true,labels_pred,verbose = False):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels_pred)
    # return purity
    purity =  np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
    
    if verbose:
        print(f"purity = {purity}")
        
    return purity

from sklearn.metrics import cluster 
def normalized_mutual_info_score(
    labels_true,
    labels_pred,
    verbose = False):
    norm_mut_info =  cluster.normalized_mutual_info_score(labels_true=labels_true,labels_pred=labels_pred)
    
    if verbose:
        print(f"norm_mut_info = {norm_mut_info}")
        
    return norm_mut_info

def adjusted_rand_score(
    labels_true,
    labels_pred,
    verbose = False):
    adjusted_rand_score =  cluster.adjusted_rand_score(labels_true=labels_true,labels_pred=labels_pred)
    
    if verbose:
        print(f"adjusted_rand_score = {adjusted_rand_score}")
        
    return adjusted_rand_score

import clustering_ml as clu
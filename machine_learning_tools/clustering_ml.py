
from collections import Counter
from matplotlib import pyplot as plt
from os import sys
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.metrics import cluster 
from sklearn.preprocessing import StandardScaler
import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import time


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

#from machine_learning_tools import clustering_ml as clu
def closet_k_neighbors_from_hierarchical_clustering(
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
        linkage="ward"
    )

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
    
        


# --------------- metrics for evaluating clusters -------------
def purity_score(labels_true,labels_pred,verbose = False):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels_pred)
    # return purity
    purity =  np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
    
    if verbose:
        print(f"purity = {purity}")
        
    return purity

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

# ----------------------- gmm utilitiex (6/30) -------------------------




sys.path.append("/notebooks/Neurosignal_Final/PRML/")
sys.path.append("/neuron_mesh_tools/Neurosignal_Final/PRML/")
#from prml.rv import VariationalGaussianMixture


# =------------- Functions to Help with Plotting -------------- #

def plot_BIC_and_Likelihood(gmm_data,
                       fig_width=12,
                       fig_height=5,
                           title_suffix = None):
    """
    Pupose
    
    """
    if title_suffix is None:
        title_suffix = ''
    
    cluster_values = list(gmm_data.keys())
    cluster_labels = "Number of Clusters"
    
    
    fig1,_ = plt.subplots(1,2)

    fig1 = mu.plot_graph(
        title = "Average Log Likelihood Of Data vs. Number of Clusters\n" + title_suffix,
        y_values = [gmm_data[K]["log_likelihood"] for K in cluster_values],
        x_values = cluster_values,
        x_axis_label = cluster_labels,
        y_axis_label = "Average Log Likelihood of Data (Per Sample)",
        figure = fig1,
        ax_index = 0,
    )


    fig1 = mu.plot_graph(
        title = "BIC vs. Number of Clusters\n" + title_suffix,
        y_values = [gmm_data[K]["bic_value"] for K in cluster_values],
        x_values = cluster_values,
        x_axis_label = cluster_labels,
        y_axis_label = "BIC (Bayesian Information Criterion)",
        figure = fig1,
        ax_index = 1
    )
    
    fig1.set_tight_layout(True)
    fig1.set_size_inches(fig_width, fig_height)
    return fig1



# ----------------- Functions for analysis ------------------- #
def gmm_analysis(X_train,
                possible_K = list(range(2,8)),
                 reg_covar = 0.00001,
                init_params = "kmeans",
                covariance_type = "full",
                 pca_obj = None,
                 scaler_obj = None,
                 column_titles = None,
                 model_type="mixture", #other type is "variational"
                 verbose=True
                ):

    """
    Purpose: Will perform gmm analysis for a specified different
    number of clusters and save the models and relevant data for further analysis
    
    """
    if column_titles is None:
        columns_picked = X_train.columns
    else:
        columns_picked = column_titles
        
        
    if type(X_train) == pd.DataFrame:
        X_train = X_train.to_numpy()
        

    scaled_original_results = dict()
    
    

    for K in possible_K:
        if verbose:
            print(f"\n\n------Working on clusters K={K}-----")
        st_time = time.time()
        scaled_original_results[K] = dict()

        reg_covar_local = copy.copy(reg_covar)
        #1) Training the GMM
        while reg_covar_local <= 0.1:
            try:
                if model_type == "mixture":
                    if verbose:
                        print("Using mixture model")
                    gmm = mixture.GaussianMixture(n_components=K, 
                                                  covariance_type=covariance_type,
                                                 reg_covar=reg_covar,
                                                 init_params=init_params)
                elif model_type == "variational":
                    if verbose:
                        print("Using variational model")
                    raise Exception("Currently not implemented")
                    #gmm = VariationalGaussianMixture(n_components=K)
                else:
                    print("Not right model")
                    raise Exception(f"The gmm model was not picked as mixture or variational : {model_type}")
                
                
                gmm.fit(X_train)
                
            except Exception as e:
                print(f"Exception occured = {str(e)}")
                print(f"Errored on gmm for reg_cov = {reg_covar_local}")
                reg_covar_local = reg_covar_local*10
            else:
                break

        if reg_covar_local >= 1:
            raise Exception(f"No gmm converged and reg_cov was {reg_covar_local}")

        if model_type == "mixture":
            bic_value = gmm.bic(X_train)
            average_log_likelihood_train = gmm.score(X_train)
            current_means = gmm.means_
        else:
            bic_value = 0
            average_log_likelihood_train = 0
            current_means = gmm.mu

        

        # Getting the Average Log likelihood:
        
        scaled_original_results[K]["model"] = gmm
        scaled_original_results[K]["log_likelihood"] = average_log_likelihood_train
        scaled_original_results[K]["bic_value"] = bic_value
        scaled_original_results[K]["reg_covar"] = reg_covar_local

        
        
        if not pca_obj is None:
            if verbose:
                print("reversing the pca transformation")
            current_means = pca_obj.inverse_transform(current_means)
        
        if not scaler_obj is None:
            if verbose:
                print("reversing the normalizing transformation")
            current_means = scaler_obj.inverse_transform(current_means)
        
        recovered_means = pd.DataFrame(current_means)
        recovered_means.columns = columns_picked

        scaled_original_results[K]["recovered_means"] = recovered_means
        
        if verbose:
            if model_type == "mixture":
                print(f"Convergence status = {gmm.converged_}")
            print(f"Total time for GMM = {time.time() - st_time}")

    return scaled_original_results

def gmm_classification(gmm_model,curr_data,
                       classification="hard",
                       verbose=True,
                       return_counts=True,
                      ):
    """
    Purpose: Will use the gaussian model passed to 
    classify the data points as to which 
    cluster they belong
    
    """
    if type(gmm_model) == mixture.GaussianMixture:
        probs = gmm_model.predict_proba(curr_data)
    elif type(gmm_model) == VariationalGaussianMixture:
        raise Exception("Currently not implemented")
        #probs = gmm_model.classify_proba(curr_data.to_numpy())
    else:
        raise Exception(f"The gmm model was not a mixture or Variational model: {type(gmm_model)}")
    
    
    if classification == "soft":
        count_values = np.sum(probs,axis=0)
    elif classification == "hard":
        gmm_class = np.argmax(probs,axis=1)
        counter_obj = Counter(gmm_class)
        count_values = []
        for clust_idx in range(gmm_model.n_components):
            if clust_idx in counter_obj.keys():
                count_values.append(counter_obj[clust_idx])
            else:
                count_values.append(0)
        count_values = np.array(count_values)
    if verbose:
        sorted_cluster_values = np.flip(np.argsort(count_values))
        print(f"Classification: {dict([(k,np.round(count_values[k],2)) for k in sorted_cluster_values])}")
    
    return count_values

def gmm_hard_classify(
    model,
    df,
    classes_as_str = False,
    verbose = False,
    ):
    if type(df) == pd.DataFrame:
        df = df.to_numpy()
    probs = model.predict_proba(df)
    gmm_class = np.argmax(probs,axis=1)
    
    if classes_as_str:
        gmm_class = [f"gmm{k}" for k in gmm_class]
    
    if verbose:
        print(f"# of classes:")
        cl,c = np.unique(gmm_class,return_counts=True)
        idx = np.flip(np.argsort(c))
        for i in idx:
            print(f"Class {cl[i]}: {c[i]}")
    
    return gmm_class


def category_classifications(model,labeled_data,
                                       return_dataframe=True,
                                       verbose = False,
                                       classification_types = ["hard","soft"]):
    total_hard = []
    total_soft = []


    labeled_data_classification = dict()
    dicts_for_classif_df = []

    for c_type in classification_types:

        if verbose:
            print(f"\nclassification_type={c_type}")
        labeled_data_classification[c_type]=dict()

        for k,v in labeled_data.items():

            if verbose:
                print(f"{k}")

            curr_class = gmm_classification(model,v,classification=c_type,verbose=verbose)
            labeled_data_classification[c_type][k] = curr_class

            classifier_dict = dict()
            classifier_dict["classification"]=c_type
            classifier_dict["category"]=k
            classifier_dict["n_clusters"] = model.n_components
            classif_dict_up = dict([(f"cl_{i}",np.round(bb,1)) for i,bb in enumerate(curr_class)])
            classifier_dict.update(classif_dict_up)

            dicts_for_classif_df.append(classifier_dict)

    if return_dataframe:
        # Print out the classification Numbers in Easy to See Dataframe
        df_class = pd.DataFrame.from_dict(dicts_for_classif_df)
        df_class.style.set_caption(f"Clustering Numbers By Neuroscience Category for K = {model.n_components}")
        df_class = df_class.sort_values(by=['category'])
        #print(df_class.to_markdown())
        
        return labeled_data_classification,df_class
    else:
        return labeled_data_classification




def clustering_stats(data,clust_perc=0.80):
    """
    Will computer different statistics about the clusters 
    formed that will be later shown or plotting 
    
    
    Metrics: For each category and classification type
    1) highest_cluster identify
    2) highest_cluster_percentage
    3) n clusters needed to encompass clust_perc % of the category
    4) Purity statistic
    
    
    """
    # categories = ["Apical","Basal","Axon"]
    # classifications = ["hard","soft"]
    # clust_perc = 0.8

    classifications = list(data.keys())
    categories = list(data[classifications[0]].keys())

    stats_dict_by_classification = dict()

    for curr_classification in classifications:
        stats_dict = dict()

        total_per_cluster_by_category = [data[curr_classification][c] for c in categories]
        total_per_cluster = np.sum(total_per_cluster_by_category,axis=0)
        for curr_category in categories:
            local_stats_dict = dict()

            count_data = data[curr_classification][curr_category]



            """
            Statistics to find:
            1) The cluster with the most of that label and the % in that cluster
            2) The number of clusters needed to comprise 80% of labeled group
            3) The purity measurements

            Pseudocode: 
            1) get the total number items put in each cluster across all categories
            2) For each cluster:
            a. Multiply the perc in that cluster * (curent number in that cluster/total number in that cluster)


            """


            sorted_labels = np.flip(np.argsort(count_data))
            highest_cluster_perc = count_data[sorted_labels[0]]/np.sum(count_data)

            local_stats_dict["highest_cluster"]  = sorted_labels[0]
            local_stats_dict["highest_cluster_perc"] = highest_cluster_perc


            sorted_labels_cumsum_perc = np.cumsum(count_data[sorted_labels]/np.sum(count_data))
            perc_per_cluster = count_data/np.sum(count_data)

            n_clusters = np.digitize(clust_perc,sorted_labels_cumsum_perc)+1
            local_stats_dict[f"n_clusters_{np.floor(clust_perc*100)}"] = n_clusters

            #find the purity metric

            purity = np.sum(perc_per_cluster[total_per_cluster != 0]*count_data[total_per_cluster != 0]/total_per_cluster[total_per_cluster != 0])

            local_stats_dict["purity"] = purity

            stats_dict[curr_category] = local_stats_dict

        # measure the purity of each cluster
        max_per_cluster = cluster_purity = np.max(total_per_cluster_by_category,axis=0)
        cluster_purity = [m/t_c if t_c > 0 else 0 for m,t_c in zip(max_per_cluster,total_per_cluster)]
        
        stats_dict_by_classification[curr_classification] = dict(cluster_purity= cluster_purity,stats_dict=stats_dict)
    return stats_dict_by_classification



def cluster_stats_dataframe(labeled_data_classification):
    """
    Purpose: Just want to visualize the soft and the hard assignment (and show they are not that different)

    Pseudocode: 
    1) 

    """

    ret_stats = clustering_stats(labeled_data_classification)

    dict_for_df = [] 

    for cl_type,cl_data in ret_stats.items():
        k = len(cl_data["cluster_purity"])
        curr_stats_dict = cl_data["stats_dict"]

        for cat_name,cat_stats_dict in curr_stats_dict.items():
            cat_local_dict = dict()
            cat_local_dict["category"] = cat_name
            cat_local_dict["classification"] = cl_type
            cat_local_dict["n_clusters"] = k
            cat_local_dict.update(cat_stats_dict)
            
            dict_for_df.append(cat_local_dict)
            
    df = pd.DataFrame.from_dict(dict_for_df)
    return df.sort_values(by=['category'])



def plot_advanced_stats_per_k(advanced_stats_per_k,
                             stats_to_plot = ["highest_cluster_perc","purity"],
                              title_suffix="",
                             fig_width = 12,
                              fig_height = 5):
    """
    Purpose: plotting the highest cluster and purity as a function of k

    Pseudocode: 
    0) Get all the possible categories, n_clusters
    0) Sort by n_clusters
    1) Iterate through all the stats we want to plot
        2) Iterate through all of the categories
            -- for all n_clusters
            a. Restrict by category and n_clusters and pull down the statistic
            b. Add to list
            --
            c. Save full list in dictionary

        3) Plot the stat using the category dictionary (using the ax index id)




    """



    advanced_stats_df = pd.concat(list(advanced_stats_per_k.values()))

    unique_categories = np.unique(advanced_stats_df["category"].to_numpy())
    unique_n_clusters = np.unique(advanced_stats_df["n_clusters"].to_numpy())
    stats_to_plot = ["highest_cluster_perc","purity"]

    cluster_labels = "Number of Clusters"

    fig, _ = plt.subplots(1,len(stats_to_plot))

    for j,st in enumerate(stats_to_plot):
        st_cat_dict = dict()
        for cat in unique_categories:
            cat_list = []
            for k in unique_n_clusters:
                curr_st = advanced_stats_df.query(f"category=='{cat}' & n_clusters=={k}")[st].to_numpy()
                if len(curr_st) != 1:
                    raise Exception("Stat was not of size 1")
                cat_list.append(curr_st[0])
            st_cat_dict[cat] = cat_list

            fig = mu.plot_graph(
                title = f"{st} vs. Number of Clusters\n" + title_suffix,
                y_values = cat_list,
                x_values = unique_n_clusters,
                x_axis_label = cluster_labels,
                y_axis_label = f"{st}",
                figure = fig,
                ax_index = j,
                label=cat
            )
    fig.set_tight_layout(True)
    fig.set_size_inches(fig_width, fig_height)
    return fig



def gmm_pipeline(df,
                title_suffix=None,
                labeled_data_indices=None, 
                 category_column=None,
                 columns_picked=None,
                 possible_K = list(range(2,8)),
                 print_tables = None, #clusters will print the clustering tables fro
                 apply_normalization=True,
                 apply_pca = True,
                 pca_whiten=True,
                 plot_sqrt_eigvals=True,
                 n_components_pca = None,
                 classification_types = ["hard"],#["hard","soft"]
                 model_type = "mixture",
                 verbose=True,
                 
                ):
    """
    Will carry out all of the clustering analysis and
    advanced stats analysis on a given dataset
    
    Arguments: 
    A data table with all of the labeled data
    """
    # ------- Initializing variables --------- #
    if title_suffix is None:
        title_suffix = ""

    if labeled_data_indices is None and category_column is not None:
        category_col = df.query(f"{category_column}=={category_column}")[category_column].to_numpy()
        if len(category_col) == 0:
            labeled_data_indices = None
        else:
            unique_labels = np.unique(category_col)
            labeled_data_indices = dict([(k,np.where(category_col==k)[0]) for k in unique_labels])
            df = pu.delete_columns(df,[category_column])
    
    if print_tables is None:
        print_tables = possible_K
        
    # -------- Part 0: Preprocessing (Column restriction, Normalization, PCA) ----------- #
    if verbose:
        print(f"# -------- Part 0: Preprocessing (Column restriction, Normalization, PCA) ----------- #")
    
    if columns_picked is None:
        columns_picked = list(df.columns) 
    else:
        if verbose:
            print(f"Restricting to columns : {columns_picked}")
        df = df[columns_picked]
    
    
    # Scaling the Data
    if apply_normalization:
        if verbose:
            print(f"Applying Normalization")
            
        scaler_obj = StandardScaler()
        df_data_scaled = scaler_obj.fit_transform(df)

        #df_data_reversed = scaler.inverse_transform(df_data_scaled,copy=True)

        data_df_normalized = pd.DataFrame(df_data_scaled)
        #add on the columns
        data_df_normalized.columns = df.columns
        df = data_df_normalized
    else:
        scaler_obj = None
        
    # Applying pca to the data
    if apply_pca:
        
        if n_components_pca is None:
            n_components_pca=len(columns_picked)
            
        if verbose:
            print(f"Applying pca with {n_components_pca} components")
            
        data_analyzed = dru.pca_analysis(df.to_numpy(),
                                    n_components=n_components_pca,
                                    whiten=pca_whiten,
                                    plot_sqrt_eigvals=plot_sqrt_eigvals)
    
        if verbose:
            print(f'Explained Variance = {data_analyzed["percent_variance_explained_up_to_n_comp"]}')
            dru.plot_variance_explained(data_analyzed)
            
        df_pca = pd.DataFrame(data_analyzed["data_proj"])
        df_pca.columns = [f"PC_{j}" for j in range(n_components_pca)]
        df = df_pca
        
        pca_obj = data_analyzed["pca_obj"]
            
    else:
        pca_obj = None
    

    
    # -------- Part 1: GMM clustering with different Number of Clusters ----------- # 
    if verbose:
        print(f"# -------- Part 1: GMM clustering with different Number of Clusters ----------- # ")
    
    X_train = df
    scaled_original_results = clu.gmm_analysis(X_train,
                    scaler_obj=scaler_obj,
                    pca_obj=pca_obj,
                     possible_K = possible_K,
                    column_titles=columns_picked,
                    model_type = model_type)
    
    if model_type == "mixture":
        fig1 = clu.plot_BIC_and_Likelihood(scaled_original_results,title_suffix=title_suffix)
        mu.display_figure(fig1)

    
    
    # --------- Part 2: computing the advanced statistics on the clustering ------- #
    if verbose:
        print(f"# --------- Part 2: computing the advanced statistics on the clustering ------- # ")
    
    
    
    advanced_stats_per_k = dict()
    labeled_data = dict([(kk,df.iloc[vv]) for kk,vv in labeled_data_indices.items()])
    
    if labeled_data is not None:
        for curr_K in scaled_original_results.keys():
            if verbose:
                print(f"\n\n----Working on Advanced Statistics for n_clusters = {curr_K}----\n")

            model = scaled_original_results[curr_K]["model"]



            labeled_data_classification,df_class=clu.category_classifications(
                                        model,
                                        labeled_data,
                                        classification_types=classification_types)
            if curr_K in print_tables:
                print("Recovered Means From Clustering")
                pu.display(scaled_original_results[curr_K]["recovered_means"])
                print("\n")
                print(f"Clustering Numbers By Neuroscience Category for K = {model.n_components}")
                pu.display_df(df_class)
                print("\n")





            cl_stats_df = clu.cluster_stats_dataframe(labeled_data_classification)

            if curr_K in print_tables:
                print(f"Clustering Advanced Statistics By Neuroscience Category for K = {curr_K}")
                pu.display_df(cl_stats_df)

            column_restriction = ["category","highest_cluster_perc","purity","n_clusters"]
            cl_stats_restricted = cl_stats_df.query("classification=='hard'")[column_restriction]

            advanced_stats_per_k[curr_K] = cl_stats_restricted
        
        # -------- Part 3: Plotting the Advanced Cluster Statistics -------------- #
        if verbose:
            print(f"# -------- Part 3: Plotting the Advanced Cluster Statistics -------------- # ")
        fig_current = clu.plot_advanced_stats_per_k(advanced_stats_per_k,title_suffix=title_suffix)
        mu.display_figure(fig_current)
    
    return scaled_original_results
    






#--- from machine_learning_tools ---
from . import dimensionality_reduction_utils as dru

#--- from datasci_tools ---
from datasci_tools import matplotlib_utils as mu
from datasci_tools import networkx_utils as xu
from datasci_tools import pandas_utils as pu

from . import clustering_ml as clu
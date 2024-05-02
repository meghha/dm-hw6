"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.linalg import eigh
from scipy.cluster.vq import kmeans2
from typing import Tuple,Optional
from scipy.special import comb

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def gaussian_kernel_similarity(x,y, sigma):
    """
    Compute the similarity matrix using the Gaussian kernel.

    Parameters:
    - data: numpy array or matrix, where each row represents a data point
    - sigma: float, bandwidth parameter for the Gaussian kernel

    Returns:
    - similarity_matrix: numpy array, similarity matrix representing the graph
    """

    # Compute pairwise squared Euclidean distances
    pairwise_distances_sq = np.sum(x - y) ** 2

    # Compute similarity matrix using Gaussian kernel formula
    similarity_matrix = np.exp(-pairwise_distances_sq / (2 * sigma**2))

    return similarity_matrix


def adjusted_rand_index(labels_true, labels_pred):
    """
    Compute the Adjusted Rand Index (ARI) to measure the similarity between two clusterings.

    Parameters:
    - labels_true: numpy array, true cluster labels
    - labels_pred: numpy array, predicted cluster labels

    Returns:
    - ari: float, Adjusted Rand Index score
    """
    n = len(labels_true)
    
    # Compute the contingency matrix
    contingency_matrix = np.zeros((np.max(labels_true) + 1, np.max(labels_pred) + 1), dtype=int)
    for i in range(n):
        contingency_matrix[labels_true[i], labels_pred[i]] += 1

    # Compute the marginal sums of the contingency matrix
    a = contingency_matrix.sum(axis=1)  # Sum of rows
    b = contingency_matrix.sum(axis=0)  # Sum of columns
    c = comb(contingency_matrix, 2).sum()  # Sum of all elements squared

    # Compute the adjusted Rand index
    index = np.sum(comb(contingency_matrix, 2)) - (np.sum(comb(a, 2)) * np.sum(comb(b, 2))) / comb(n, 2)
    expected_index = (np.sum(comb(a, 2)) * np.sum(comb(b, 2))) / comb(n, 2)
    max_index = (np.sum(comb(a, 2)) + np.sum(comb(b, 2))) / 2
    ari = (index - expected_index) / (max_index - expected_index)

    return ari

def computeSSE(data, labels):
    sse = 0.0
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    return sse

def cluster_plots(data, labels, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> Tuple[Optional[NDArray[np.int32]], Optional[float], Optional[float], Optional[NDArray[np.floating]]]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """

    sigma = params_dict['sigma']
    k = params_dict['k']

    # Similarity matrix
    n_samples = data.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            similarity_matrix[i, j] = gaussian_kernel_similarity(data[i], data[j], sigma)

    # Laplacian matrix
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = eigh(laplacian_matrix)

    # KMEANS++ clustering
    _, computed_labels = kmeans2(eigenvectors[:, 1:k], k, minit='++')

    # SSE
    SSE = computeSSE(data, computed_labels)

    # Compute adjusted Rand index
    ARI = adjusted_rand_index(labels, computed_labels)

    return computed_labels, SSE, ARI, eigenvalues

def hyperparameter_study(data, labels):
    """
    Perform hyperparameter study for spectral clustering on the given data.

    Arguments:
    - data: input data array of shape (n_samples, n_features)
    - labels: true labels of the data

    Return values:
    - sigmas: Array of sigma values
    - ari_scores: Array of ARI scores for each sigma value
    - sse_scores: Array of SSE scores for each sigma value
    """
    ari_vals = []
    sse_vals = []

    sigmas = np.logspace(-1, 1, num=10)  
    k = 5  

    for sigma in sigmas:
        _, sse, ari, _ = spectral(data, labels, {'sigma': sigma, 'k': k})
        sse_vals.append(sse)
        ari_vals.append(ari)

    return sigmas, np.array(ari_vals), np.array(sse_vals)

def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    
    cluster_data = np.load('question1_cluster_data.npy')
    cluster_labels = np.load('question1_cluster_labels.npy')
    
    data_subset = cluster_data[:1000]
    labels_subset = cluster_labels[:1000]
    
    # perform hyperparameter study for first 1000 points
    sigmas, ari_values, sse_values = hyperparameter_study(data_subset, labels_subset)
    
    
    
    # ARI vs sigma
    plt.figure(figsize=(7, 5))
    plt.plot(sigmas, ari_values, color='b')
    plt.title('ARI vs Sigma')
    plt.xlabel('Sigma')
    plt.ylabel('ARI')
    plt.grid(True)
    plt.xscale('log') 
    plt.savefig('ARI vs Sigma')
    plt.tight_layout()
    plt.close()
    
    # SSE vs sigma
    plt.figure(figsize=(7, 5))
    plt.plot(sigmas, sse_values, color='r')
    plt.title('SSE vs Sigma')
    plt.xlabel('Sigma')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.xscale('log') 
    plt.savefig('SSE vs Sigma')
    plt.close()
    
    
    # After hyperparameter study
    best_sigma = 0.1
    best_k = 5
    
    plots_values={}

    # The best paramaeters from hyperparameter study are considered
    
    for i in [0,1,2,3,4]:
        
        data_slice = cluster_data[i * 1000: (i + 1) * 1000]
        labels_slice = cluster_labels[i * 1000: (i + 1) * 1000]
        computed_labels, sse, ari, eig_values = spectral(data_slice, labels_slice, {'sigma': best_sigma, 'k': best_k})
        groups[i] = {"sigma": best_sigma, "ARI": ari, "SSE": sse}
        plots_values[i] = {"computed_labels": computed_labels, "ARI": ari, "SSE": sse,"eig_values":eig_values} 
        
    highest_ari = -1
    best_dataset_index = None

    for i, group_info in plots_values.items():
        if group_info['ARI'] > highest_ari:
            highest_ari = group_info['ARI']
            best_dataset_index = i
            
    
    # Plot the clusters for the dataset with the highest ARI

    plt.figure(figsize=(7, 5))
    plot_ARI = plt.scatter(cluster_data[best_dataset_index * 1000: (best_dataset_index + 1) * 1000, 0], 
                cluster_data[best_dataset_index * 1000: (best_dataset_index + 1) * 1000, 1], 
                c=plots_values[best_dataset_index]["computed_labels"], cmap='viridis')
    plt.title(f'Clustering for highest ARI | k:{best_k}')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.savefig('Clustering with highest ARI.png')
    plt.close()
    
    # Find the dataset with the lowest SSE
    lowest_sse = float('inf')
    best_dataset_index_sse = None
    for i, group_info in plots_values.items():
        if group_info['SSE'] < lowest_sse:
            lowest_sse = group_info['SSE']
            best_dataset_index_sse = i
    
    # Plot the clusters for the dataset with the lowest SSE
    plt.figure(figsize=(8, 6))
    plot_SSE = plt.scatter(cluster_data[best_dataset_index_sse * 1000: (best_dataset_index_sse + 1) * 1000, 0], 
                cluster_data[best_dataset_index_sse * 1000: (best_dataset_index_sse + 1) * 1000, 1], 
                c=plots_values[best_dataset_index_sse]["computed_labels"], cmap='viridis')
    plt.title(f'Clustering for lowest SSE | k:{best_k}')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Clustering with lowest SSE.png') 
    
    plt.close()
    
    
    # Plot of the eigenvalues (smallest to largest) as a line plot for all datasets
    plt.figure(figsize=(8, 6))
    
    for i, group_info in plots_values.items():
        plot_eig = plt.plot(np.sort(group_info["eig_values"]), label=f'Dataset {i+1}')
    
    plt.title('Eigen Values')
    
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Eigen Values.png') 
    plt.close()
    
 
    

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"] #{}

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    #plot_ARI = plt.scatter([1,2,3], [4,5,6])
    #plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    #plot_eig = plt.plot([1,2,3], [4,5,6])
    answers["eigenvalue plot"] = plot_eig

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    
    ari_values = [group_info["ARI"] for group_info in groups.values()]
    mean_ari = np.mean(ari_values)
    std_dev_ari = np.std(ari_values)

    # A single float
    answers["mean_ARIs"] = mean_ari

    # A single float
    answers["std_ARIs"] = std_dev_ari
    
    sse_values = [group_info["SSE"] for group_info in groups.values()]
    mean_sse = np.mean(sse_values)
    std_sse = np.std(sse_values)

    # A single float
    answers["mean_SSEs"] = mean_sse

    # A single float
    answers["std_SSEs"] = std_sse

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.spatial.distance import cdist
from typing import Tuple,Optional
from scipy.special import comb

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def kMeansScratchCode(data, k, max_iter=100, tol=1e-4):
    rows, cols = data.shape

    # random centroid initialization
    centroids = data[np.random.choice(rows, k, replace=False)]
    
    for iter in range(max_iter):

        # calculate distance of each point from centroid
        dists = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

        # cluster assignment
        clusters = np.argmin(dists, axis=1)

        new_centroids = np.zeros((k, cols))

        for i in range(k):
            
            cluster_values = data[clusters == i]
            if cluster_values.size > 0:
                new_centroids[i] = cluster_values.mean(axis=0)
            else:

                # reinitialize to a random centroid if cluster is empty
                new_centroids[i] = data[np.random.choice(rows, 1, replace=False)].flatten()

        # convergence
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids

    return clusters, centroids


def sse(data, labels, centroids):
    """
    Calculate the Sum of Squared Errors (SSE) for given data and centroids.
    
    Parameters:
        data (np.ndarray): The dataset.
        labels (np.ndarray): Array of cluster labels for each data point.
        centroids (np.ndarray): Array of centroids, one for each cluster.
    
    Returns:
        float: The calculated SSE.
    """
    k = len(centroids)
    SSE = 0
    for i in range(k):
        cluster_data = data[labels == i]
        if cluster_data.size > 0:
            SSE += np.sum((cluster_data - centroids[i])**2)
    return SSE


def adjusted_rand_index(true_labels, pred_labels):
    
    n = len(true_labels)
    categories = np.unique(true_labels)
    clusters = np.unique(pred_labels)

    # contingency table
    contingency = np.array([[np.sum((true_labels == category) & (pred_labels == cluster)) for cluster in clusters] for category in categories])
    
    sumC = np.sum([comb(n_c, 2) for n_c in np.sum(contingency, axis=1)])
    sumK = np.sum([comb(n_k, 2) for n_k in np.sum(contingency, axis=0)])
    sumComb = np.sum([comb(n_ij, 2) for n_ij in contingency.flatten()])
    total_comb = comb(n, 2)
    expected_comb = sumC * sumK / total_comb
    max_comb = (sumC + sumK) / 2
    
    if total_comb == expected_comb:  # Prevent division by zero
        return 0.0
    else:
        ARI = (sumComb - expected_comb) / (max_comb - expected_comb)
        return ARI


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

    # similarity matrix
    dists = cdist(data, data, 'sqeuclidean')
    W = np.exp(-dists / (2 * sigma**2))

    # diagonal matrix
    D = np.diag(W.sum(axis=1))

    # Laplacian matrix
    L = D - W

    # Computation of eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    V = eigenvectors[:, :k]
    
    # kMeans
    computed_labels, centroids = kMeansScratchCode(V, k)

   # Computation of SSE
    SSE = sse(V, computed_labels, centroids)


    # Compute ARI
    ARI = adjusted_rand_index(labels, computed_labels)

    return computed_labels, SSE, ARI, eigenvalues



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
    sse = []
    ari = []
    max_ari = 0
    min_sse = 0

    sigmaMaxARI = 0
    sigmaMinSSE = 0

    sigma = np.linspace(0.1, 10, 10)
    k = 5
    data = np.load('question1_cluster_data.npy')[:5000]
    labels = np.load('question1_cluster_labels.npy')[:5000]

    params_dict = {}
    for i in range(len(sigma)):
        params_dict['sigma'] = sigma[i]
        params_dict['k'] = k
        computed_labels, SSE, ARI, eigenvalues = spectral(data[:1000], labels[:1000], params_dict)
        if i==0:
            min_sse = SSE
            sigmaMinSSE = sigma[i]
        elif SSE < min_sse:
            min_sse = SSE
            sigmaMinSSE = sigma[i]
        if ARI > max_ari:
            max_ari = ARI
            sigmaMaxARI = sigma[i]
        ari.append(ARI)
        sse.append(SSE)


    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    
    max_ari_data = None
    max_ari_labels = None
    min_sse_data = None
    min_sse_labels = None

    final_sigma = sigmaMaxARI
    eigenvalues = np.array([])
    
    for i in range(5):
        params_dict['sigma'] = final_sigma
        params_dict['k'] = k
        index_start = 1000 * i
        index_end = 1000 * (i + 1)
        computed_labels, SSE, ARI, eigenvalue = spectral(data[index_start:index_end], labels[index_start:index_end], params_dict)
        groups[i] = {"sigma": float(final_sigma), "ARI": float(ARI), "SSE": float(SSE)}
        eigenvalues = np.append(eigenvalues, eigenvalue, axis=0)
        
        if i==0:
            min_sse = SSE
            min_sse_data = data[index_start:index_end]
            min_sse_labels = computed_labels
            max_ari = ARI
            max_ari_data = data[index_start:index_end]
            max_ari_labels = computed_labels
        
        if ARI > max_ari:
            max_ari = ARI
            max_ari_data = data[index_start:index_end]
            max_ari_labels = computed_labels

        if SSE < min_sse:
            min_sse = SSE
            min_sse_data = data[index_start:index_end]
            min_sse_labels = computed_labels

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.
    
    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"]

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


    sigmas = np.array([group["sigma"] for group in groups.values()])
    ARIs = np.array([group["ARI"] for group in groups.values()])
    SSEs = np.array([group["SSE"] for group in groups.values()])

    
    plot_ARI = plt.scatter(max_ari_data[:, 0], max_ari_data[:, 1], c=max_ari_labels, cmap='viridis', s=25)
    plt.title('Clusters with Largest ARI')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # plt.savefig('Clusters with Largest ARI')
    plt.close()

    

    answers["cluster scatterplot with largest ARI"] = plot_ARI


    
    
    plot_SSE = plt.scatter(min_sse_data[:, 0], min_sse_data[:, 1], c=min_sse_labels, cmap='viridis', s=25)
    plt.title('Clusters with Smallest SSE')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # plt.savefig('Clusters with Smallest SSE')
    plt.close()

    

    answers["cluster scatterplot with smallest SSE"] = plot_SSE


    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    plot_eig = plt.plot(np.sort(eigenvalues), linestyle='-')
    plt.title("Sorted Eigen Values")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    # plt.savefig('Sorted Eigen Values')
    plt.close()

    answers["eigenvalue plot"] = plot_eig


    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = float(np.mean(ARIs))

    # A single float
    answers["std_ARIs"] = float(np.std(ARIs))

    # A single float
    answers["mean_SSEs"] = float(np.mean(SSEs))

    # A single float
    answers["std_SSEs"] = float(np.std(SSEs))

    return answers





# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)

"""
@author Bennet Wittelsbach, based on code from Alejandro Molina
"""

import numpy as np
from typing import Callable, List, Tuple

from sklearn.cluster import KMeans  # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore
from spflow.base.learning.splitting.base import preprocess, split_data_by_clusters
from spflow.base.learning.context import Context


def get_split_rows_KMeans(
    n_clusters: int = 2, pre_proc: str = None, ohe: bool = False, seed: int = 17
) -> Callable:
    """Wrapper function for the KMeans splitting procedure in the clustering step of LearnSPN.

    The sklearn implementation of the KMeans algorithm is used here. Further documentation can be found at
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html.


    Arguments:
        n_clusters:
            Number of clusters for the KMeans algorithm.
        pre_proc:
            Options for preprocessing the data.
        ohe:
            Flag for applying one-hot encoding to data columns containing discrete values.
        seed:
            The random seed for sklearn.

    Returns:
        A function handler for the KMeans clustering procedure with the given arguments that will be used in LearnSPN.
    """

    def split_rows_KMeans(
        local_data: np.ndarray, context: Context, scope: list[int]
    ) -> List[Tuple[np.ndarray, List[int], float]]:
        """Function for KMeans clustering with parameters set by outer function and arguments set during the learning process.

        Arguments:
            local_data:
                A (2-dimensional) numpy array, holding the data to be clustered.
            context:
                Context needed for preprocessing.
            scope:
                The list of scopes to cluster and split.

        Returns:
            A list of result tuples, each result is a cluster. Each entry contains the cluster's data,
            the scope of the cluster and the proportion of the cluster's data in relation to the total data.
        """
        data = preprocess(local_data, context, pre_proc, ohe)

        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(
            data[:, scope]
        )  # WARNING: I've changed predict(data) to predict(data[:, scope]) to be able to append indices without using them during clustering. still needs testing if this doesn't break anything

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_KMeans


def get_split_rows_GMM(
    n_clusters: int = 2,
    pre_proc: str = None,
    ohe: bool = False,
    seed: int = 17,
    max_iter: int = 100,
    n_init: int = 2,
    covariance_type: str = "full",
) -> Callable:
    """Wrapper function for the Gaussian Mixture Model splitting procedure in the clustering step of LearnSPN.

    The sklearn implementation of the KMeans algorithm is used here. Further documentation can be found at
    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html.

    Arguments:
        n_clusters:
            Number of clusters for the KMeans algorithm.
        pre_proc:
            Options for preprocessing the data.
        ohe:
            Flag for applying one-hot encoding to data columns containing discrete values.
        seed:
            The random seed for sklearn.
        max_iter:
            Number of iterations of the Expectation-Maximization algorithm used in the GMM procedure.
        n_init:
            Number of initializations/repetitions of the EM algorithm. The best result will be kept.
        covariance_type:
            Type of covariance parameters to use:
                - full
                    each component has its own general covariance matrix
                - tied
                    all components share the same general covariance matrix
                - diag
                    each component has its own diagonal covariance matrix
                - spherical
                    each component has its own single variance

    Returns:
        A function handler for the Gaussian Mixture Model clustering procedure with the given arguments that will be used in LearnSPN.
    """

    def split_rows_GMM(
        local_data: np.ndarray, context: Context, scope: List[int]
    ) -> List[Tuple[np.ndarray, List[int], float]]:
        """Function for GMM clustering with parameters set by outer function and arguments set during the learning process.

        Arguments:
            local_data:
                A (2-dimensional) numpy array, holding the data to be clustered.
            context:
                Context needed for preprocessing.
            scope:
                The list of scopes to cluster and split.

        Returns:
            A list of result tuples, each result is a cluster. Each entry contains the cluster's data,
            the scope of the cluster and the proportion of the cluster's data in relation to the total data.
        """
        data = preprocess(local_data, context, pre_proc, ohe)

        estimator = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=seed,
        )

        clusters = estimator.fit_predict(
            data[:, scope]
        )  # WARNING: I've changed fit(data)/predict(data) to fit/predict(data[:, scope]) to be able to append indices without using them during clustering. still needs testing if this doesn't break anything

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_GMM


if __name__ == "__main__":
    kmeans = get_split_rows_KMeans(n_clusters=2)
    gmm = get_split_rows_GMM(n_clusters=2, max_iter=20, covariance_type="full")

    from sklearn.datasets import make_blobs  # type: ignore

    cluster_centers = [(-5, -5), (5, 5)]
    cluster_stdev = [3, 8]
    # cluster_centers = np.multiply(cluster_centers, 100)
    # cluster_stdev = np.multiply(cluster_stdev, 100)
    total_samples = 10000
    scope = [0, 1]
    data, label = make_blobs(
        n_samples=total_samples,
        centers=cluster_centers,
        cluster_std=cluster_stdev,
        n_features=2,
        random_state=17,
    )
    data = np.concatenate(
        (data, label.reshape(-1, 1), np.arange(0, total_samples, 1).reshape(-1, 1)), axis=1
    )  # np.arange(0, total_samples, 1).reshape(-1, 1)

    print(data)

    kmeans_cluster = kmeans(data, None, scope)
    print(kmeans_cluster)
    kmeans_cluster1 = kmeans_cluster[0][0]
    kmeans_cluster2 = kmeans_cluster[1][0]
    kmeans_accuracy = (
        len(kmeans_cluster1[kmeans_cluster1[:, 2] == 0, :])
        + len(kmeans_cluster2[kmeans_cluster2[:, 2] == 1, :])
    ) / total_samples

    gmm_cluster = gmm(data, None, scope)
    gmm_cluster1 = gmm_cluster[0][0]
    gmm_cluster2 = gmm_cluster[1][0]
    gmm_accuracy = (
        len(gmm_cluster1[gmm_cluster1[:, 2] == 0, :])
        + len(gmm_cluster2[gmm_cluster2[:, 2] == 1, :])
    ) / total_samples

    # from scipy.stats import multivariate_normal
    # from matplotlib import cm
    # X = np.linspace(-20, 20, 400)
    # Y = np.linspace(-20, 20, 400)
    # XY = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis= 1)
    # XY = np.array(np.meshgrid(X, Y)).reshape(-1, 2)
    # print(XY.shape)
    # gmm_cluster1_probs = multivariate_normal.pdf(XY, np.mean(gmm_cluster1[:, [0, 1]], axis=0), np.cov(gmm_cluster1[:, [0, 1]], rowvar=False))
    # gmm_cluster2_probs = multivariate_normal.pdf(XY, np.mean(gmm_cluster2[:, [0, 1]], axis=0), np.cov(gmm_cluster2[:, [0, 1]], rowvar=False))
    # print(gmm_cluster1_probs, gmm_cluster2_probs.shape)
    # gmm_clusters_diff = (gmm_cluster1_probs - gmm_cluster2_probs).reshape(400, 400)

    print(
        f"accuracy - kmeans: {kmeans_accuracy}, gmm: {gmm_accuracy}"
    )  # this is not entirely correct, determine accuracy or confusion matrix via ID of data points

    from matplotlib import pyplot as plt  # type: ignore

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].scatter(data[label == 0, 0], data[label == 0, 1], s=5)
    axes[0].scatter(data[label == 1, 0], data[label == 1, 1], s=5)
    axes[0].set_title("Ground truth")
    axes[1].scatter(kmeans_cluster1[:, 0], kmeans_cluster1[:, 1], s=5)
    axes[1].scatter(kmeans_cluster2[:, 0], kmeans_cluster2[:, 1], s=5)
    axes[1].set_title("K-Means")
    axes[2].scatter(gmm_cluster1[:, 0], gmm_cluster1[:, 1], s=5)
    axes[2].scatter(gmm_cluster2[:, 0], gmm_cluster2[:, 1], s=5)
    axes[2].set_title("GMM")
    # levels = np.arange(-1.0, 1.01, 0.2)
    # cset = axes[2].contourf(X, Y, gmm_clusters_diff, levels=levels)# , cmap=cm.get_cmap(cm.RdYlBu, len(levels) - 1))
    # fig.colorbar(cset, ax=axes[2])
    fig.tight_layout()
    plt.show()
    plt.close()

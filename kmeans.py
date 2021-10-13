# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


# Given the data points and cluster centers, this function finds the closest cluster for each point
def get_closest_cluster(points, cluster_centers, dist_metric="L2"):
    # When dist_metric is L1, we apply L1 norm
    if dist_metric == "L1":
        order = 1
    else:
        # Default distance metric for np.linalg.norm is L2
        order = None
    return np.argmin([np.linalg.norm(points - x, axis=1, ord=order) for x in cluster_centers], axis=0)


# Given the data points and their cluster assignments, this function adjusts the cluster centers
def get_new_cluster_centers(points, cluster_assn, cluster_centers, dist_metric="L2"):
    # List of numpy arrays representing each pixel in each cluster
    data = [points[np.where(cluster_assn == x)] for x in range(len(cluster_centers))]

    # Only re-adjusting cluster centers for non-empty clusters, removing empty ones
    # by checking if len(data belonging to cluster x) > 0

    # If distance metric is L1, we use the k-median function
    if dist_metric == "L1":
        new_cluster_centers = np.array([x.median(axis=0) for x in data if len(x) > 0])
    else:
        # Otherwise, we use k-means function
        new_cluster_centers = np.array([x.mean(axis=0) for x in data if len(x) > 0])
    return new_cluster_centers


# Runs k-means algorithm on pixels
def kmeans_run(points, K, dist_metric="L2"):
    # np.random.seed(42)

    # Random initialization of K data points to be cluster centers
    old_cluster_centers = points[np.random.choice(len(points), K, replace=False)]

    # Assign each pixel to its nearest cluster
    cluster_assns = get_closest_cluster(points, old_cluster_centers, dist_metric)

    # Number of iterations, i
    i = 1

    # Do until cluster centers do not change
    while True:

        # For each cluster, calculate the new cluster centroid to be the mean
        cluster_centers = get_new_cluster_centers(points, cluster_assns, old_cluster_centers, dist_metric)

        # Assign each point to the closest cluster center
        cluster_assns = get_closest_cluster(points, cluster_centers, dist_metric)

        # Check if the cluster centers have changed; If yes, algorithm has reached convergence
        if (cluster_centers.size == old_cluster_centers.size) and (
                np.equal(cluster_centers, old_cluster_centers).sum() == old_cluster_centers.size):
            break
        i += 1

        # Storing cluster centers in variable old_cluster_centers for comparison later
        old_cluster_centers = cluster_centers

    # Only returning non-empty clusters. If an empty cluster was deleted, send a message to the user;
    k = len(cluster_centers)
    if K > k:
        print(f"Current value of K producing empty cluster. Reducing K from {K} to {k}")
    return cluster_assns, cluster_centers, i, k

# Function gets the within-cluster point scatter for a cluster assignment
def get_total_wcps(pixels, cluster_assns, cluster_centers, dist_metric="L2"):
    if dist_metric == "L1":
        order = 1
    else:
        order=None
    pixel_centroids = np.apply_along_axis(lambda x: cluster_centers[x], axis=0, arr=cluster_assns)
    wcps = np.linalg.norm(pixels - pixel_centroids, axis=1, ord=order).sum()
    return wcps


# Does the full run of K-Means for multiple values of K and dist_metric
# Does 5 iterations of K-Means for a single k,L1 combination to try  5 different random initial centroids
def kmeans_full(pixels, K=[2, 4, 8, 16], num_iter=5, dist_metric=["L1", "L2"]):
    # dictionary keeping track of metrics for each k-means run
    results = {
        "rand_init": [],
        "K": [],
        "num_iters": [],
        "time": [],
        "cluster_assns": [],
        "cluster_centers": [],
        "dist_metric": [],
        "Total_WCPS": [],
        "dist_from_orig": []
    }

    for dist in dist_metric:
        for k in K:
            for randinit in range(num_iter):
                # print(k, dist)

                # Recording the time it takes for each k, L1 iteration
                start = time.perf_counter()
                cluster_assns, cluster_centers, iters, k_changed = kmeans_run(pixels, k, dist_metric)
                end = time.perf_counter()
                length_time = end - start
                # within-cluster point scatter
                wcps = get_total_wcps(pixels, cluster_assns, cluster_centers, dist_metric)

                # calculating the distance from the cluster assignment to the original image
                if dist == "L1":
                    dist_from_image = (np.linalg.norm(cluster_centers[cluster_assns] - pixels, ord=1, axis=1)).sum()
                else:
                    dist_from_image = (np.linalg.norm(cluster_centers[cluster_assns] - pixels, axis=1)).sum()

                # Storing metrics in results dictionary
                results['rand_init'].append(randinit)
                results['K'].append(k_changed)
                results['num_iters'].append(iters)
                results['time'].append(length_time)
                results['cluster_assns'].append(cluster_assns)
                results['cluster_centers'].append(cluster_centers)
                results['dist_metric'].append(dist)
                results['Total_WCPS'].append(wcps)
                results['dist_from_orig'].append(dist_from_image)

    return results
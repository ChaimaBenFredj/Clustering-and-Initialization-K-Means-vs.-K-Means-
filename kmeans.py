import numpy as np

# We start by randomly picking k points from the dataset to use as the starting centroids. 
    
def initialize_random(X, k, random_state=None):
    """Picking k random dots from X and returning them as the initial guesses for where the cluster centers are"""
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()

# Assigning each point to its nearest centroid, returns one cluster index per point
def assign_clusters(X, centroids): 
    """  First step: Assignment; Finding the index of the nearest centroid for each point,
     The distances line computes the distance from every point to every centroid all at once. """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k):
    """ Second Step: Update; Move each centroid to the mean of its assigned points.
    If a cluster is empty (no points assigned), keep the old centroid in place by returning NaN """
    new_centroids = np.zeros((k, X.shape[1]))
    for j in range(k):
        points_in_cluster = X[labels == j]
        if len(points_in_cluster) == 0:           
            new_centroids[j] = np.nan    # If the cluster is empty — signal with NaN
        else:
            new_centroids[j] = points_in_cluster.mean(axis=0)
    return new_centroids

def compute_inertia(X, labels, centroids):
    """ The Inertia measures how tight the clusters are: it's the sum of squared distances from each point to its centroid 
        (lower means better) """
    total = 0.0
    for j in range(len(centroids)):
        points = X[labels == j]
        if len(points) > 0:
            total += np.sum((points - centroids[j]) ** 2)
    return total



def kmeans(X, k, max_iters=300, tol=1e-4, random_state=None):
    """ The kmeans algorithm: running the full k-means loop — assign, update, repeat until convergence.
         returns the cluster labels, final centroid positions, inertia score, and number of iterations. """
    centroids = initialize_random(X, k, random_state=random_state) #Random Initialization
    labels = None

    for i in range(max_iters):  #max_iters: maximum number of iterations
        new_labels = assign_clusters(X, centroids) #Assignment

        new_centroids = update_centroids(X, new_labels, k)  #Update

        # Handling empty clusters by keeping old centroid
        empty = np.any(np.isnan(new_centroids), axis=1)
        new_centroids[empty] = centroids[empty]

        # Convergence check 
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels = new_labels

        if centroid_shift < tol:  #tol: convergence tolerance (stops if centroids move less than this)
            break

    inertia = compute_inertia(X, labels, centroids)
    return labels, centroids, inertia, i + 1 #n_iters: number of iterations run


# Quick test 

if __name__ == "__main__":
    from datasets import load_synthetic
    import matplotlib.pyplot as plt

    X, y_true = load_synthetic()

    print("Running k-means on the synthetic dataset\n")

    labels, centroids, inertia, n_iters = kmeans(X, k=5, random_state=0)

    print(f"  Cluster sizes : {[int((labels == j).sum()) for j in range(5)]}")

    # Plotting the result 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=10, alpha=0.5)
    axes[0].set_title("True Clusters Structure")

    axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=10, alpha=0.5)
    axes[1].scatter(centroids[:, 0], centroids[:, 1],
                    c='black', marker='X', s=200, zorder=5, label='Centroids')
    axes[1].set_title(f"K-Means Result  (inertia={inertia:.1f})")
    axes[1].legend()

    plt.suptitle("K-Means from Scratch", fontsize=13)
    plt.tight_layout()
    plt.savefig("kmeans_result.png", dpi=120)
    plt.show()
    print("\nSaved: kmeans_result.png")

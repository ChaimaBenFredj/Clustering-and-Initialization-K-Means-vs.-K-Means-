import numpy as np
from kmeans import assign_clusters, update_centroids, compute_inertia

# The Kmeans++ Init picks centroids one by one, each new one chosen toward points far from the already chosen ones
def initialize_plus_plus(X, k, random_state=None):
    """ unlike the random init in kmeans, we don't pick all centroids at once: each new centroid is chosen 
    with higher probability if it's far from the existing ones, so they end up spread across the data. """
    rng = np.random.RandomState(random_state)
    n = len(X)
    centroids = []

    # Picking the first centroid uniformly at random (the 1st centroid is still random because thee is no other way, it has to be random) 
    first_idx = rng.randint(0, n)
    centroids.append(X[first_idx].copy())

    # Picking the remaining centroids 
    for _ in range(k - 1):

        centroid_array = np.array(centroids)          
        dists = np.linalg.norm(
            X[:, np.newaxis] - centroid_array, axis=2
        )
        # squared distance from each point to its nearest centroid                                          
        D_squared = np.min(dists, axis=1) ** 2        

        # farther points get higher probability
        probs = D_squared / D_squared.sum()

        # Pick the next centroid
        next_idx = rng.choice(n, p=probs)
        centroids.append(X[next_idx].copy())

    return np.array(centroids)


def kmeans_plus_plus(X, k, max_iters=300, tol=1e-4, random_state=None):
    """ The K-means++ algorithm.
    Identical to the k-means algorithm, the diffirence is in the initialization step."""
    centroids = initialize_plus_plus(X, k, random_state=random_state) #Initialization (main difference with kmeans)

    labels = None

    for i in range(max_iters):
        new_labels = assign_clusters(X, centroids) #Assignment

        new_centroids = update_centroids(X, new_labels, k) #Update

        empty = np.any(np.isnan(new_centroids), axis=1) # Handling empty clusters by keeping old centroid
        new_centroids[empty] = centroids[empty]

        # Convergence check 
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels = new_labels

        if centroid_shift < tol:
            break

    inertia = compute_inertia(X, labels, centroids)
    return labels, centroids, inertia, i + 1


# Quick test : side by side comparison

if __name__ == "__main__":
    from datasets import load_synthetic
    from kmeans import kmeans
    import matplotlib.pyplot as plt

    X, y_true = load_synthetic()
    K = 5

    print("Single trial comparison (same random seed)\n")
    print("─" * 40)

    labels_km,   centroids_km,   inertia_km,   iters_km   = kmeans(X, k=K, random_state=0)
    labels_kpp,  centroids_kpp,  inertia_kpp,  iters_kpp  = kmeans_plus_plus(X, k=K, random_state=0)

    print(f"  K-Means    →  inertia: {inertia_km:.2f}  |  iterations: {iters_km}  |  cluster sizes: {sorted([int((labels_km==j).sum()) for j in range(K)], reverse=True)}")
    print(f"  K-Means++  →  inertia: {inertia_kpp:.2f}  |  iterations: {iters_kpp}  |  cluster sizes: {sorted([int((labels_kpp==j).sum()) for j in range(K)], reverse=True)}")
    print(f"\n  Inertia improvement: {((inertia_km - inertia_kpp) / inertia_km * 100):.1f}%")

    # Plotting side by side 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=10, alpha=0.5)
    axes[0].set_title("Ground Truth")

    axes[1].scatter(X[:, 0], X[:, 1], c=labels_km, cmap='tab10', s=10, alpha=0.5)
    axes[1].scatter(centroids_km[:, 0], centroids_km[:, 1],
                    c='black', marker='X', s=200, zorder=5)
    axes[1].set_title(f"K-Means\nInertia: {inertia_km:.1f}")

    axes[2].scatter(X[:, 0], X[:, 1], c=labels_kpp, cmap='tab10', s=10, alpha=0.5)
    axes[2].scatter(centroids_kpp[:, 0], centroids_kpp[:, 1],
                    c='black', marker='X', s=200, zorder=5)
    axes[2].set_title(f"K-Means++\nInertia: {inertia_kpp:.1f}")

    plt.suptitle("Single Run: K-Means vs K-Means++", fontsize=13)
    plt.tight_layout()
    plt.savefig("kmeans_vs_plus_plus.png", dpi=120)
    plt.show()
    print("\nSaved: kmeans_vs_plus_plus.png")


    # Showing the initialization difference 
    rng_km  = np.random.RandomState(0)
    init_km = X[rng_km.choice(len(X), size=K, replace=False)]
    init_kpp = initialize_plus_plus(X, K, random_state=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, init, title in zip(axes,
                                [init_km, init_kpp],
                                ["Random Init (K-Means)", "Smart Init (K-Means++)"]):
        ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=10, alpha=0.5)
        ax.scatter(init[:, 0], init[:, 1],
                   c=range(K), cmap='tab10', s=300, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.set_title(title)

    plt.suptitle("Initialization: Where do the starting centroids land?", fontsize=13)
    plt.tight_layout()
    plt.savefig("initialization_comparison.png", dpi=120)
    plt.show()
    print("Saved: initialization_comparison.png")

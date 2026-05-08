import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# generating sythetic data, useful for testing before touching real data
def load_synthetic(n_samples=1500, n_clusters=5, random_state=42):
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=0.8,
        random_state=random_state
    )
    return X, y_true


# using sklearn's built-in digits instead of full MNIST, good enough for our project
def load_digits_dataset():
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target
    X = X / 16.0
    return X, y


# Plotting scatter plot for the synthetic data, colors by cluster label
def plot_synthetic(X, y_true):
    k = len(np.unique(y_true))
    cmap = plt.cm.get_cmap('tab10', k)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X[:, 0], X[:, 1],
                          c=y_true, cmap=cmap,
                          vmin=-0.5, vmax=k-0.5,
                          s=15, alpha=0.6)
    cbar = plt.colorbar(scatter, ticks=range(k))
    cbar.set_label('True cluster')
    plt.title("Synthetic Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig("synthetic_dataset.png", dpi=120)
    plt.show()


# Doing a quick visual check to make sure the images loaded correctly
def plot_digits_samples(X, y, n=20):
    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(8, 8), cmap='gray')
        ax.set_title(str(y[i]), fontsize=8)
        ax.axis('off')
    plt.suptitle("Digits Dataset Sample Images (8x8 pixels)", fontsize=12)
    plt.tight_layout()
    plt.savefig("digits_samples.png", dpi=120)
    plt.show()
    


if __name__ == "__main__":
    print("Synthetic Dataset:")
    X_syn, y_syn = load_synthetic()
    print(f"Shape: {X_syn.shape}  |  Clusters: {len(np.unique(y_syn))}")
    plot_synthetic(X_syn, y_syn)

    print("\nDigits Dataset (high-dimensional):")
    X_dig, y_dig = load_digits_dataset()
    print(f"Shape: {X_dig.shape}  |  Classes: {sorted(np.unique(y_dig))}")
    plot_digits_samples(X_dig, y_dig)


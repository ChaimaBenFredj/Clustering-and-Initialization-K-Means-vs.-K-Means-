import numpy as np
import matplotlib.pyplot as plt
from datasets import load_synthetic, load_digits_dataset
from kmeans import kmeans, initialize_random
from kmeans_plus_plus import kmeans_plus_plus, initialize_plus_plus

# Choosing a global style to be used
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'legend.fontsize':   9,
    'figure.dpi':        130,
})

KM_COLOR  = '#e05c5c'   # red color for k-means
KPP_COLOR = '#4a90d9'   # blue color for k-means++
N_TRIALS  = 50


def run_trials(X, k, n_trials=50):
    inertias_km, inertias_kpp = [], []
    for seed in range(n_trials):
        _, _, inertia_km,  _ = kmeans(X, k=k, random_state=seed)
        _, _, inertia_kpp, _ = kmeans_plus_plus(X, k=k, random_state=seed)
        inertias_km.append(inertia_km)
        inertias_kpp.append(inertia_kpp)
    return np.array(inertias_km), np.array(inertias_kpp)


# Figure 1 shows Initialization Comparison for the Synthetic dataset
# It shows visually how the two methods place starting centroids

def figure1_initialization(X, y_true):
    K = 5
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # true Clusters
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=8, alpha=0.5)
    axes[0].set_title("(a) True Clusters Structure")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")

    # Random init
    init_km = initialize_random(X, K, random_state=7)
    axes[1].scatter(X[:, 0], X[:, 1], c='lightgray', s=8, alpha=0.4)
    axes[1].scatter(init_km[:, 0], init_km[:, 1],
                    c=range(K), cmap='tab10', s=350, marker='*',
                    edgecolors='black', linewidths=0.8, zorder=5)
    axes[1].set_title("(b) Random Initialization")
    axes[1].set_xlabel("Feature 1")

    # K-means++ init
    init_kpp = initialize_plus_plus(X, K, random_state=7)
    axes[2].scatter(X[:, 0], X[:, 1], c='lightgray', s=8, alpha=0.4)
    axes[2].scatter(init_kpp[:, 0], init_kpp[:, 1],
                    c=range(K), cmap='tab10', s=350, marker='*',
                    edgecolors='black', linewidths=0.8, zorder=5)
    axes[2].set_title("(c) K-Means++ Initialization")
    axes[2].set_xlabel("Feature 1")

    plt.suptitle("Figure 1 — Initialization Strategies on Synthetic Data", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("fig1_initialization.png", dpi=130, bbox_inches='tight')
    plt.show()
    print("Saved: fig1_initialization.png")


# Fihure 2 shows Clustering Result Comparison for the Synthetic datset
# it shows the Best vs worst run for each algorithm

def figure2_clustering_results(X, y_true):
    K = 5

    # Collecting results over many seeds
    results_km, results_kpp = [], []
    for seed in range(N_TRIALS):
        labels, centroids, inertia, _ = kmeans(X, k=K, random_state=seed)
        results_km.append((inertia, labels, centroids))
        labels, centroids, inertia, _ = kmeans_plus_plus(X, k=K, random_state=seed)
        results_kpp.append((inertia, labels, centroids))

    best_km  = min(results_km,  key=lambda x: x[0])
    worst_km = max(results_km,  key=lambda x: x[0])
    best_kpp = min(results_kpp, key=lambda x: x[0])
    worst_kpp= max(results_kpp, key=lambda x: x[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    configs = [
        (axes[0, 0], best_km,   KM_COLOR,  "K-Means — Best run"),
        (axes[0, 1], worst_km,  KM_COLOR,  "K-Means — Worst run"),
        (axes[1, 0], best_kpp,  KPP_COLOR, "K-Means++ — Best run"),
        (axes[1, 1], worst_kpp, KPP_COLOR, "K-Means++ — Worst run"),
    ]

    for ax, (inertia, labels, centroids), color, title in configs:
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=8, alpha=0.4)
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c='black', marker='X', s=180, zorder=5)
        ax.set_title(f"{title}\nInertia: {inertia:.1f}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    plt.suptitle("Figure 2 — Best and Worst Clustering Results (50 trials)", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig2_best_worst.png", dpi=130, bbox_inches='tight')
    plt.show()
    print("Saved: fig2_best_worst.png")


# Figure 3 shows the inertia Distribution for both datasets
# The main statistical comparison

def figure3_inertia_distribution(km_syn, kpp_syn, km_dig, kpp_dig):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    datasets = [
        (axes[0], km_syn,  kpp_syn,  "Synthetic Dataset  (k=5)"),
        (axes[1], km_dig,  kpp_dig,  "Digits Dataset  (k=10)"),
    ]

    for ax, km, kpp, title in datasets:
        bins = np.linspace(
            min(km.min(), kpp.min()) * 0.97,
            max(km.max(), kpp.max()) * 1.03,
            28
        )
        ax.hist(km,  bins=bins, alpha=0.65, color=KM_COLOR,  label='K-Means')
        ax.hist(kpp, bins=bins, alpha=0.65, color=KPP_COLOR, label='K-Means++')
        ax.axvline(km.mean(),  color='darkred', linestyle='--', linewidth=1.8,
                   label=f'KM mean:  {km.mean():.0f}')
        ax.axvline(kpp.mean(), color='navy',    linestyle='--', linewidth=1.8,
                   label=f'KM++ mean: {kpp.mean():.0f}')
        ax.set_xlabel("Inertia")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend()

    plt.suptitle("Figure 3 — Distribution of Inertia over 50 Trials", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig3_inertia_distribution.png", dpi=130, bbox_inches='tight')
    plt.show()
    print("Saved: fig3_inertia_distribution.png")


# Figure 4: Variability Boxplot for both datasets

def figure4_boxplot(km_syn, kpp_syn, km_dig, kpp_dig):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    datasets = [
        (axes[0], km_syn,  kpp_syn,  "Synthetic Dataset  (k=5)"),
        (axes[1], km_dig,  kpp_dig,  "Digits Dataset  (k=10)"),
    ]

    for ax, km, kpp, title in datasets:
        bp = ax.boxplot(
            [km, kpp],
            tick_labels=['K-Means', 'K-Means++'],
            patch_artist=True,
            medianprops=dict(color='black', linewidth=2),
            flierprops=dict(marker='o', markersize=4, alpha=0.5)
        )
        bp['boxes'][0].set_facecolor(KM_COLOR)
        bp['boxes'][1].set_facecolor(KPP_COLOR)
        ax.set_ylabel("Inertia")
        ax.set_title(title)

        # Annotate standard deviation
        for i, data in enumerate([km, kpp], start=1):
            ax.text(i, data.max() * 1.01, f'σ={data.std():.1f}',
                    ha='center', fontsize=9, color='gray')

    plt.suptitle("Figure 4 — Inertia Variability over 50 Trials", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig4_boxplot.png", dpi=130, bbox_inches='tight')
    plt.show()
    print("Saved: fig4_boxplot.png")


# Figure 5 shows the effect of k on inertia on the synthetic dataset
# how does performance gap change with more clusters?

def figure5_effect_of_k(X):
    k_values = [2, 3, 4, 5, 6, 8, 10]
    mean_km, mean_kpp = [], []
    std_km,  std_kpp  = [], []

    for k in k_values:
        km, kpp = run_trials(X, k=k, n_trials=50)
        mean_km.append(km.mean());   std_km.append(km.std())
        mean_kpp.append(kpp.mean()); std_kpp.append(kpp.std())

    mean_km  = np.array(mean_km);  std_km  = np.array(std_km)
    mean_kpp = np.array(mean_kpp); std_kpp = np.array(std_kpp)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(k_values, mean_km,  color=KM_COLOR,  marker='o', label='K-Means')
    ax.fill_between(k_values, mean_km - std_km, mean_km + std_km,
                    color=KM_COLOR, alpha=0.15)

    ax.plot(k_values, mean_kpp, color=KPP_COLOR, marker='o', label='K-Means++')
    ax.fill_between(k_values, mean_kpp - std_kpp, mean_kpp + std_kpp,
                    color=KPP_COLOR, alpha=0.15)

    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean inertia (50 trials)")
    ax.set_title("Figure 5 — Mean Inertia vs k  (shaded = ±1 std)")
    ax.legend()
    ax.set_xticks(k_values)

    plt.tight_layout()
    plt.savefig("fig5_effect_of_k.png", dpi=130, bbox_inches='tight')
    plt.show()
    print("Saved: fig5_effect_of_k.png")



# Main

if __name__ == "__main__":
    #Loading datasets
    X_syn, y_syn = load_synthetic()
    X_dig, y_dig     = load_digits_dataset()

    #Running 50 trials for each dataset
    km_syn,  kpp_syn  = run_trials(X_syn, k=5,  n_trials=N_TRIALS)
    km_dig,  kpp_dig  = run_trials(X_dig, k=10, n_trials=N_TRIALS)

    figure1_initialization(X_syn, y_syn)
    figure2_clustering_results(X_syn, y_syn)
    figure3_inertia_distribution(km_syn, kpp_syn, km_dig, kpp_dig)
    figure4_boxplot(km_syn, kpp_syn, km_dig, kpp_dig)
    figure5_effect_of_k(X_syn)    
    print("\nAll figures are saved")


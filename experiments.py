import numpy as np
import matplotlib.pyplot as plt
from datasets import load_synthetic, load_digits_dataset
from kmeans import kmeans
from kmeans_plus_plus import kmeans_plus_plus

# This function runs both algorithms n_trials times (we chose 50) with different seeds and collects the inertia scores
def run_trials(X, k, n_trials=50):
    inertias_km  = []
    inertias_kpp = []

    for seed in range(n_trials):
        _, _, inertia_km,  _ = kmeans(X, k=k, random_state=seed)
        _, _, inertia_kpp, _ = kmeans_plus_plus(X, k=k, random_state=seed)

        inertias_km.append(inertia_km)
        inertias_kpp.append(inertia_kpp)

    return np.array(inertias_km), np.array(inertias_kpp)

# printing a stats table that contains mean, std, min and max for both algorithms on our datasets

def print_summary(name, inertias_km, inertias_kpp):
    print(f"\n{'─'*50}")
    print(f"  Dataset: {name}")
    print(f"{'─'*50}")
    print(f"  {'Metric':<22} {'K-Means':>12} {'K-Means++':>12}")
    print(f"  {'─'*46}")
    print(f"  {'Mean inertia':<22} {inertias_km.mean():>12.2f} {inertias_kpp.mean():>12.2f}")
    print(f"  {'Std deviation':<22} {inertias_km.std():>12.2f} {inertias_kpp.std():>12.2f}")
    print(f"  {'Min inertia':<22} {inertias_km.min():>12.2f} {inertias_kpp.min():>12.2f}")
    print(f"  {'Max inertia':<22} {inertias_km.max():>12.2f} {inertias_kpp.max():>12.2f}")
    print(f"  {'Worst / Best ratio':<22} {inertias_km.max()/inertias_km.min():>12.2f} {inertias_kpp.max()/inertias_kpp.min():>12.2f}")
    improvement = (inertias_km.mean() - inertias_kpp.mean()) / inertias_km.mean() * 100
    print(f"\n  Mean inertia improvement (++): {improvement:.1f}%")

# Plotting a histogram and a sorted line plot to show how inertia scores are distributed across the 50 trials
def plot_inertia_distribution(inertias_km, inertias_kpp, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    bins = np.linspace(
        min(inertias_km.min(), inertias_kpp.min()) * 0.98,
        max(inertias_km.max(), inertias_kpp.max()) * 1.02,
        30
    )
    ax.hist(inertias_km,  bins=bins, alpha=0.6, color='tomato',    label='K-Means')
    ax.hist(inertias_kpp, bins=bins, alpha=0.6, color='steelblue', label='K-Means++')
    ax.axvline(inertias_km.mean(),  color='darkred', linestyle='--', linewidth=1.5,
               label=f'KM mean: {inertias_km.mean():.0f}')
    ax.axvline(inertias_kpp.mean(), color='navy',    linestyle='--', linewidth=1.5,
               label=f'KM++ mean: {inertias_kpp.mean():.0f}')
    ax.set_xlabel("Inertia")
    ax.set_ylabel("Count (out of 50 trials)")
    ax.set_title("Distribution of Inertia Values")
    ax.legend(fontsize=8)

    ax = axes[1]
    sorted_km  = np.sort(inertias_km)
    sorted_kpp = np.sort(inertias_kpp)
    trials = np.arange(1, len(sorted_km) + 1)

    ax.plot(trials, sorted_km,  color='tomato',    marker='o', markersize=3,
            linewidth=1.5, label='K-Means')
    ax.plot(trials, sorted_kpp, color='steelblue', marker='o', markersize=3,
            linewidth=1.5, label='K-Means++')
    ax.set_xlabel("Trial rank (sorted by inertia)")
    ax.set_ylabel("Inertia")
    ax.set_title("Sorted Inertia per Trial")
    ax.legend(fontsize=8)

    plt.suptitle(f"Stability Comparison — {title}  (50 trials)", fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()
    print(f"Saved: {filename}")

# Plot to see inertia improvements: uses % improvement so both datasets are comparable. dots above 0 mean k-means++ won that trial

def plot_improvement(km_syn, kpp_syn, km_dig, kpp_dig, filename):
    imp_syn = (km_syn - kpp_syn) / km_syn * 100
    imp_dig = (km_dig - kpp_dig) / km_dig * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, imp, color, title in zip(
        axes,
        [imp_syn, imp_dig],
        ['tomato', 'steelblue'],
        ['Synthetic (k=5)', 'Digits (k=10)']
    ):
        trials = np.arange(1, len(imp) + 1)
        ax.scatter(trials, imp, color=color, alpha=0.7, s=40)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.axhline(imp.mean(), color=color, linestyle='-', linewidth=1.5,
                   label=f'mean: {imp.mean():.1f}%')
        ax.set_xlabel("Trial")
        ax.set_title(title)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("K-Means++ improvement (%)")
    plt.suptitle("K-Means++ improvement over K-Means — per trial", fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()
    print(f"Saved: {filename}")


if __name__ == "__main__":

    N_TRIALS = 50

    print("Running trials on Synthetic dataset...")
    X_syn, _ = load_synthetic()
    km_syn, kpp_syn = run_trials(X_syn, k=5, n_trials=N_TRIALS)
    print_summary("Synthetic (k=5)", km_syn, kpp_syn)
    plot_inertia_distribution(km_syn, kpp_syn,
                              title="Synthetic Dataset",
                              filename="trials_synthetic.png")

    print("\nRunning trials on Digits dataset...")
    X_dig, _ = load_digits_dataset()
    km_dig, kpp_dig = run_trials(X_dig, k=10, n_trials=N_TRIALS)
    print_summary("Digits (k=10)", km_dig, kpp_dig)
    plot_inertia_distribution(km_dig, kpp_dig,
                              title="Digits Dataset",
                              filename="trials_digits.png")

    plot_improvement(
        km_syn, kpp_syn,
        km_dig, kpp_dig,
        filename="improvement_boxplot.png"
    )

    print("\nAll experiments done!")
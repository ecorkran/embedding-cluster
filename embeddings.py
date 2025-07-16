import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*")

import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from clusters import cluster_embeddings
import matplotlib as mpl
from matplotlib import colormaps
import matplotlib.gridspec as gridspec

def get_file_paths(src_pattern):
    return sorted(glob.glob(src_pattern))

def read_files(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            yield f.read()

def embed_texts(texts, model):
    return model.encode(texts)

def dry_run(paths):
    print(f"{len(paths)} files matched:")
    for p in paths:
        print(os.path.abspath(p))

def plot_similarity_histogram(similarities, ax, cbar_ax, n_bins=100):
    # Plot a shaded histogram with percentiles
    hist_cmap = plt.get_cmap('Blues')
    n, bins, patches = ax.hist(
        similarities, bins=n_bins,
        color=hist_cmap(0.7), edgecolor=hist_cmap(0.9), alpha=0.9
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    norm_hist = mpl.colors.Normalize(vmin=min(similarities), vmax=max(similarities))
    for c, p in zip(bin_centers, patches):
        fc = hist_cmap(norm_hist(c))
        p.set_facecolor(fc)
        p.set_edgecolor(hist_cmap(min(norm_hist(c)+0.2, 1.0)))
    ax.set_xlabel('Cosine similarity')
    ax.set_ylabel('Number of pairs')
    ax.set_title('Similarity Histogram')
    sm_hist = mpl.cm.ScalarMappable(norm=norm_hist, cmap=hist_cmap)
    sm_hist.set_array([])
    cbar = plt.colorbar(sm_hist, cax=cbar_ax, orientation='vertical')
    cbar.set_ticks([])

def cluster_plot(X, graph_labels):

    # 1) Project to 2D
    tsne = TSNE(n_components=2, random_state=42, init="pca")
    X2 = tsne.fit_transform(X)

    # 2) Scatter plot, coloring by cluster
    plt.figure()
    plt.scatter(
        X2[:, 0], X2[:, 1],
        c=graph_labels,
        s=50,  # point size
        edgecolor='k'  # black outline so colors stand out
    )
    plt.title("t-SNE of Task Embeddings\ncolored by Graph-based Cluster")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(label="Cluster ID")
    plt.tight_layout()
    plt.show()


def print_similarity_table(similarities, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.65, 0.96, 0.05)
    print("\nSimilarity thresholds:")
    print(f"{'Threshold':>10} | {'Count':>6} | {'Percentile':>10}")
    print("-" * 32)
    total = len(similarities)
    for t in thresholds:
        count = np.sum(similarities >= t)
        pct = 100.0 * count / total if total else 0
        print(f"   {t:>6.2f}   | {count:>6} | {pct:>9.2f}%")

def print_percentile_table(similarities, percentiles=None):
    if percentiles is None:
        percentiles = [50, 75, 80, 85, 90, 95, 99]
    print("\nSimilarity percentiles:")
    print(f"{'Percentile':>10} | {'Similarity':>10}")
    print("-" * 25)
    for p in percentiles:
        val = np.percentile(similarities, p)
        print(f"   {p:>6.0f}%   |   {val:>8.4f}")

def multi_cluster_plot(X, cluster_results, similarities, annotate_points=False, threshold=None):
    # Compute 2D projection once
    tsne = TSNE(n_components=2, random_state=42, init="pca")
    X2 = tsne.fit_transform(X)
    labels_dict = {
        "Graph": cluster_results["graph_labels"],
        "Agglomerative": cluster_results["agg_labels"],
        "HDBSCAN": cluster_results["hdb_labels"],
    }
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 6, height_ratios=[3, 2], width_ratios=[5, 0.3, 5, 0.3, 5, 0.3])

    for i, (name, labels) in enumerate(labels_dict.items()):
        ax = plt.subplot(gs[0, i*2])
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_to_count = dict(zip(unique_labels, counts))
        # Identify singleton clusters
        singleton_labels = set(lbl for lbl, count in label_to_count.items() if count == 1)
        non_singleton_labels = [lbl for lbl in unique_labels if lbl not in singleton_labels]
        n_non_singleton = len(non_singleton_labels)
        if n_non_singleton > 0:
            color_map = plt.get_cmap('viridis', n_non_singleton)
            label_to_color = {lbl: color_map(i) for i, lbl in enumerate(non_singleton_labels)}
        else:
            label_to_color = {}
        # Singleton clusters: all neutral light grey
        grey = (0.82, 0.82, 0.82, 1.0)
        for lbl in singleton_labels:
            label_to_color[lbl] = grey
        # Assign colors and edgecolors to each point
        point_colors = np.array([label_to_color[lbl] for lbl in labels])
        edgecolors = np.array([
            'none' if lbl in singleton_labels else 'k'
            for lbl in labels
        ])
        scatter = ax.scatter(
            X2[:, 0], X2[:, 1],
            c=point_colors,
            s=50,
            edgecolors=edgecolors,
        )
        ax.set_title(f"{name} Clusters")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        # Colorbar for non-singleton clusters only (placed next to each plot)
        if n_non_singleton > 0:
            norm = mpl.colors.Normalize(vmin=0, vmax=n_non_singleton-1)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=color_map)
            cax = fig.add_subplot(gs[0, i*2+1])
            cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
            cbar.set_ticks([])
        if annotate_points:
            for (x, y, lbl) in zip(X2[:, 0], X2[:, 1], labels):
                ax.text(x, y, str(lbl), fontsize=8, ha='center', va='center', color='black', alpha=0.7)
        if annotate_points:
            print(f"{name} cluster sizes:")
            for lbl, count in zip(unique_labels, counts):
                print(f"  Cluster {lbl}: {count} points")

    # Bottom row: similarity histogram via helper
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist_cbar = fig.add_subplot(gs[1, 1])
    plot_similarity_histogram(similarities, ax_hist, ax_hist_cbar)
    # Add threshold display in the new subplot area
    ax_border = fig.add_subplot(gs[1, 2:6])
    ax_border.axis('off')
    ax_border.text(
        0.00, 0.98, f"Clustering threshold: {threshold}",
        ha='left', va='top', fontsize=10,
        transform=ax_border.transAxes
    )
    plt.tight_layout()
    plt.show()

def compute_embeddings(paths):
    texts = list(read_files(paths))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.vstack(embed_texts(texts, model))

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("src_pattern", help="Source file glob pattern, e.g. mydir/tasks.*.md")
    parser.add_argument("--dry-run", action="store_true", help="Only list files to be processed and exit")
    parser.add_argument("--cluster-threshold", type=float, default=0.75, help="Clustering similarity threshold (default: 0.75)")
    parser.add_argument("--debug", action="store_true", help="Show cluster labels on plot and print cluster sizes")
    args = parser.parse_args()

    paths = get_file_paths(args.src_pattern)
    if args.dry_run:
        print("Files to be processed:")
        for path in paths:
            print(path)
        return

    x = compute_embeddings(paths)

    # Compute cosine similarity
    sims = cosine_similarity(x)
    upper = sims[np.triu_indices_from(sims, k=1)]

    print(f"{len(paths)} files, {len(upper)} unique pairs")
    print("First 5 similarities:", upper[:5])
    print_similarity_table(upper)
    print_percentile_table(upper)

    print("\nClustering summary (Graph / Agglomerative / HDBSCAN):")
    cluster_results = cluster_embeddings(x, threshold=args.cluster_threshold, verbose=True)
    multi_cluster_plot(x, cluster_results, upper, annotate_points=args.debug, threshold=args.cluster_threshold)
    # Optionally, you could further process or print cluster_results here

if __name__ == "__main__":
    main()

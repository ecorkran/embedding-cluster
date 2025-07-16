import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import hdbscan
import pandas as pd

def summarize(labels, name):
    # ignore “noise” for graph & agg (they won’t have -1)
    unique, counts = np.unique(labels, return_counts=True)
    # treat -1 as noise
    noise = counts[unique == -1][0] if -1 in unique else 0
    cluster_sizes = counts[unique != -1]
    return {
        "method": name,
        "n_clusters": len(cluster_sizes),
        "min_size": int(cluster_sizes.min()) if len(cluster_sizes) else 0,
        "max_size": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
        "avg_size": float(cluster_sizes.mean()) if len(cluster_sizes) else 0.0,
        "n_noise/singleton": int(noise + np.sum(cluster_sizes == 1))
    }

def cluster_embeddings(X, threshold=0.75, verbose=True):
    # 1. Compute pairwise sims & distance matrix
    sims = cosine_similarity(X)
    dists = 1 - sims
    N = X.shape[0]

    # 2. Graph-based clustering
    G = nx.Graph()
    G.add_nodes_from(range(N))
    i_inds, j_inds = np.triu_indices_from(sims, k=1)
    for i, j in zip(i_inds, j_inds):
        if sims[i, j] >= threshold:
            G.add_edge(i, j)
    graph_comps = list(nx.connected_components(G))
    graph_labels = np.zeros(N, dtype=int) - 1
    for lbl, comp in enumerate(graph_comps):
        for idx in comp:
            graph_labels[idx] = lbl

    # 3. Agglomerative clustering
    agg = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - float(threshold),
        n_clusters=None
    )
    agg_labels = agg.fit_predict(dists)

    # 4. HDBSCAN clustering
    hdb = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=2,
        cluster_selection_epsilon=0.1
    )
    hdb_labels = hdb.fit_predict(X)

    # 5. Summarize
    results = [
        summarize(graph_labels, "Graph"),
        summarize(agg_labels,   "Agglomerative"),
        summarize(hdb_labels,   "HDBSCAN"),
    ]
    df = pd.DataFrame(results)
    if verbose:
        print(df.to_markdown(index=False))
    return {
        "graph_labels": graph_labels,
        "agg_labels": agg_labels,
        "hdb_labels": hdb_labels,
        "summary": df
    }

def sweep_thresholds(
    X: np.ndarray,
    *,
    t_min: float = 0.40,
    t_max: float = 0.95,
    steps: int  = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (thresholds, n_clusters, n_singletons) for a range of graph thresholds.
    Useful for picking a good cutoff visually.
    """
    thresholds   = np.linspace(t_min, t_max, steps)
    n_clusters   = np.zeros_like(thresholds, dtype=int)
    n_singletons = np.zeros_like(thresholds, dtype=int)

    sims = cosine_similarity(X)
    ii, jj = np.triu_indices_from(sims, k=1)
    n = X.shape[0]

    for k, t in enumerate(thresholds):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from((i, j) for i, j in zip(ii, jj) if sims[i, j] >= t)
        comps = list(nx.connected_components(G))
        n_clusters[k]   = len(comps)
        n_singletons[k] = sum(1 for c in comps if len(c) == 1)

    return thresholds, n_clusters, n_singletons

def sweep_agglomerative(
    X: np.ndarray,
    *,
    t_min: float = 0.40,
    t_max: float = 0.95,
    steps: int  = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep over similarity thresholds for agglomerative clustering.
    Returns (thresholds, n_clusters, n_singletons).
    """
    thresholds   = np.linspace(t_min, t_max, steps)
    n_clusters   = np.zeros_like(thresholds, dtype=int)
    n_singletons = np.zeros_like(thresholds, dtype=int)

    sims = cosine_similarity(X)
    dists = 1 - sims

    for k, t in enumerate(thresholds):
        agg = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - float(t),
            n_clusters=None
        )
        labels = agg.fit_predict(dists)
        unique, counts = np.unique(labels, return_counts=True)
        mask = unique != -1
        # count valid clusters (excluding noise)
        n_clusters[k] = int(np.count_nonzero(mask))
        # count singleton clusters among valid labels
        n_singletons[k] = int(np.count_nonzero(counts[mask] == 1))

    return thresholds, n_clusters, n_singletons
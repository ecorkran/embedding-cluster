# Embedding Cluster Analysis

Semantic clustering tool for finding patterns in AI-generated code review feedback.

## What it does

- Embeds text using sentence-transformers (all-MiniLM-L6-v2)
- Computes pairwise cosine similarity
- Clusters using three methods: graph-based, agglomerative, HDBSCAN
- Visualizes with t-SNE projections
- Includes threshold sweep for parameter tuning

## Why

Built to identify duplicate/similar issues in large sets of AI-generated code review comments. Reduces noise, surfaces patterns.

## Usage
```bash
python main.py "data/tasks.*.md" --cluster-threshold 0.75 --sweep --dump
```

## Tech

Python, NumPy, scikit-learn, HDBSCAN, NetworkX, sentence-transformers, matplotlib

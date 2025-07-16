import textwrap
from collections import defaultdict
import numpy as np
from typing import List

def preview(text: str, max_len: int = 120) -> str:
    """Return a cleaned single-line preview of a markdown task file."""
    lines = text.splitlines()

    # 1) Strip optional YAML front-matter
    if lines and lines[0].strip() == "---":
        try:
            end = lines.index("---", 1)
            lines = lines[end + 1:]
        except ValueError:
            pass  # malformed front-matter; keep all lines

    # 2) Drop blank lines & Markdown headings consisting only of '#' chars
    for ln in lines:
        ln = ln.strip()
        if not ln or (ln.startswith("#") and ln.strip("#").strip() == ""):
            continue
        # first good line
        return (ln[:max_len] + "…") if len(ln) > max_len else ln

    # fallback to compressed text
    fallback = " ".join(text.split())[:max_len]
    return fallback + ("…" if len(fallback) == max_len else "")


def dump_clusters(
    labels: np.ndarray,
    texts: List[str],
    *,
    title: str,
    min_size: int = 2,
    show_full: bool = False,
):
    """Pretty print clusters in the console."""
    groups = defaultdict(list)
    for idx, lbl in enumerate(labels):
        groups[lbl].append(idx)

    print(f"\n{title}")
    for lbl, idxs in groups.items():
        if len(idxs) < min_size:
            continue
        print(f"\nCluster {lbl}  (size = {len(idxs)})")
        for i in idxs:
            if show_full:
                # indent the full file for easy reading
                print(textwrap.indent(texts[i].rstrip(), "    "))
                print("    ———")
            else:
                print(f" • [{i:02}] {preview(texts[i])}")

    # Singletons
    singles = [i for i, lbl in enumerate(labels) if len(groups[lbl]) == 1]
    if singles:
        print("\nSingleton tasks")
        for i in singles:
            if show_full:
                print(textwrap.indent(texts[i].rstrip(), "    "))
                print("    ———")
            else:
                print(f" • [{i:02}] {preview(texts[i])}")

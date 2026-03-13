from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from nscluster.utils import ensure_dir


def cluster_features(cfg: dict):
    out_root = Path(cfg["out_root"])
    feat_dir = out_root / "features"
    feat_path = feat_dir / "features.npy"
    meta_path = feat_dir / "meta.csv"
    if not feat_path.exists() or not meta_path.exists():
        raise FileNotFoundError("features not found. Run features step first.")

    clusters_dir = out_root / "clusters"
    ensure_dir(clusters_dir)

    meta = pd.read_csv(meta_path)
    n = len(meta)

    # 读 memmap
    # 维度从文件大小推断不方便，这里用 np.load(mmap_mode) 读取
    feats = np.load(feat_path, mmap_mode="r")
    if feats.shape[0] != n:
        raise RuntimeError(f"meta rows {n} != features rows {feats.shape[0]}")

    pca_dim = int(cfg["pca_dim"])
    method = cfg["cluster_method"].lower()

    # PCA fit（随机化 SVD 更快）
    print(f"[cluster] PCA -> {pca_dim} dims")
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=42)

    # 为了内存稳，先抽样 fit（2w 可以全 fit；更大数据建议抽样）
    fit_n = min(n, 200000)  # 大数据抽 20w
    sample_idx = np.random.RandomState(42).choice(n, size=fit_n, replace=False) if fit_n < n else np.arange(n)
    pca.fit(np.asarray(feats[sample_idx], dtype=np.float32))

    # 分批 transform
    X = np.zeros((n, pca_dim), dtype=np.float32)
    bs = 65536
    for s in tqdm(range(0, n, bs), desc="[cluster] PCA transform"):
        e = min(n, s + bs)
        X[s:e] = pca.transform(np.asarray(feats[s:e], dtype=np.float32))

    if method == "kmeans":
        k = int(cfg["kmeans_k"])
        kbs = int(cfg.get("kmeans_batch_size", 4096))
        print(f"[cluster] MiniBatchKMeans: k={k}, batch={kbs}")
        km = MiniBatchKMeans(
            n_clusters=k,
            batch_size=kbs,
            random_state=42,
            n_init="auto",
        )
        km.fit(X)
        labels = km.labels_.astype(int)
        # 到最近中心距离（可用于挑“低置信”样本）
        dists = np.min(km.transform(X), axis=1).astype(np.float32)

    elif method == "hdbscan":
        try:
            import hdbscan
        except Exception as e:
            raise RuntimeError("hdbscan not installed. pip install hdbscan") from e
        print("[cluster] HDBSCAN (may be slow on large N)")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
        labels = clusterer.fit_predict(X).astype(int)
        # HDBSCAN 没有天然的“距离”，用 membership probability 近似
        probs = getattr(clusterer, "probabilities_", None)
        dists = (1.0 - probs).astype(np.float32) if probs is not None else np.zeros((n,), dtype=np.float32)
    else:
        raise ValueError(f"Unknown cluster_method: {method}")

    # 保存 assignments
    out_assign = clusters_dir / "assignments.csv"
    df = meta.copy()
    df["cluster_id"] = labels
    df["score"] = dists
    df.to_csv(out_assign, index=False, encoding="utf-8")

    # 保存统计
    stats = df.groupby("cluster_id").size().reset_index(name="count").sort_values("count", ascending=False)
    out_stats = clusters_dir / "cluster_stats.csv"
    stats.to_csv(out_stats, index=False, encoding="utf-8")

    print(f"[cluster] saved -> {out_assign}")
    print(f"[cluster] saved -> {out_stats}")
from pathlib import Path
import pandas as pd
import csv
import shutil

from nscluster.ann_io import load_annotation, dump_annotation, iter_shapes
from nscluster.utils import ensure_dir


def read_cluster_map(path: str) -> dict[int, str]:
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["cluster_id"])
            lab = row["label"].strip()
            mp[cid] = lab
    return mp


def apply_cluster_labels(cfg: dict, cluster_map_csv: str, overwrite_main: bool):
    out_root = Path(cfg["out_root"])
    label_root = Path(cfg["label_root"])
    refined_dir = out_root / "label_refined"
    ensure_dir(refined_dir)

    clusters_dir = out_root / "clusters"
    assign_path = clusters_dir / "assignments.csv"
    if not assign_path.exists():
        raise FileNotFoundError("assignments.csv not found. Run cluster step first.")

    mp = read_cluster_map(cluster_map_csv)
    df = pd.read_csv(assign_path)

    # 先把原标注整体复制一份到 refined_dir（保持目录结构）
    # 注意：如果 label_root 很大，复制可能慢；你也可以改成“只写改动文件”模式。
    print(f"[apply] copying label_root -> {refined_dir} (may take time)")
    if refined_dir.exists():
        # 避免把旧结果叠加到新结果：只删内部文件，不删目录（保险起见）
        shutil.rmtree(refined_dir)
        ensure_dir(refined_dir)
    shutil.copytree(label_root, refined_dir, dirs_exist_ok=True)

    # 分组处理：按 ann_path 聚合
    grouped = df.groupby("ann_path")

    n_updated = 0
    n_skipped = 0

    for ann_path, g in grouped:
        ann_path = Path(ann_path)
        # 找到复制后的对应文件路径
        rel = ann_path.relative_to(label_root)
        out_ann = refined_dir / rel

        if not out_ann.exists():
            # 有些 ann_path 可能不是 label_root 下的，直接跳过
            n_skipped += len(g)
            continue

        ann, fmt = load_annotation(out_ann)
        shapes = list(iter_shapes(ann))

        changed = False
        for _, row in g.iterrows():
            cid = int(row["cluster_id"])
            if cid not in mp:
                # 没有映射的簇不改（你可以把它们统一映射到 other）
                continue
            fine = mp[cid]
            si = int(row["shape_idx"])
            if si < 0 or si >= len(shapes):
                continue

            shape = shapes[si]
            if overwrite_main:
                shape["main"] = fine
                shape["label"] = fine
            else:
                # 更安全：写入 sub 字段
                shape["sub"] = fine
            changed = True
            n_updated += 1

        if changed:
            dump_annotation(out_ann, ann, fmt=fmt)

    print(f"[apply] updated instances: {n_updated}, skipped: {n_skipped}")
    print(f"[apply] refined labels dir -> {refined_dir}")
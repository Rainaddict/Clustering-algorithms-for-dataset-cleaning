from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from nscluster.dinov2_embedder import DinoV2Embedder    
from nscluster.image_crop import crop_instance_rgb
from nscluster.utils import ensure_dir, set_seed


class InstanceDataset(Dataset):
    def __init__(self, index_jsonl: Path, use_polygon_mask: bool, input_size: int):
        self.items = []
        with open(index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))
        self.use_polygon_mask = use_polygon_mask
        self.input_size = input_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rec = self.items[i]
        img = crop_instance_rgb(
            rec["image_path"],
            rec["bbox_xyxy"],
            polygon=rec.get("polygon"),
            use_polygon_mask=self.use_polygon_mask,
        )
        # img: PIL RGB
        return rec["instance_id"], rec["ann_path"], rec["shape_idx"], img


def collate_fn(batch):
    ids, ann_paths, shape_idxs, imgs = zip(*batch)
    return list(ids), list(ann_paths), list(shape_idxs), list(imgs)


def extract_features(cfg: dict):
    set_seed(42)

    out_root = Path(cfg["out_root"])
    index_jsonl = out_root / "index" / "instances.jsonl"
    if not index_jsonl.exists():
        raise FileNotFoundError(f"index not found: {index_jsonl}. Run index step first.")

    feat_dir = out_root / "features"
    ensure_dir(feat_dir)

    use_polygon_mask = bool(cfg["use_polygon_mask"])
    input_size = int(cfg["input_size"])
    device = cfg["device"]
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    model_name = cfg["model_name"]

    ds = InstanceDataset(index_jsonl, use_polygon_mask=use_polygon_mask, input_size=input_size)
    n = len(ds)
    print(f"[features] instances: {n}")

    # memmap 保存特征：float16，节省空间
    feat_path = feat_dir / "features.npy"
    meta_path = feat_dir / "meta.csv"

    # 先初始化 memmap 文件
    # DINOv2 输出维度：vitb14=768, vitl14=1024（这里动态获取）
    embedder = DinoV2Embedder(model_name=model_name, input_size=input_size, device=device)
    dim = embedder.dim
    print(f"[features] model={model_name}, dim={dim}, device={device}")

    # feats = np.memmap(feat_path, dtype=np.float16, mode="w+", shape=(n, dim))

    feats = np.lib.format.open_memmap(feat_path, dtype=np.float16, mode="w+", shape=(n, dim))

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    rows = []
    offset = 0

    with torch.no_grad():
        for ids, ann_paths, shape_idxs, imgs in tqdm(dl, desc="[features] extracting"):
            # imgs: list[PIL]
            emb = embedder.encode(imgs)  # torch [B, dim], already L2 norm
            emb_np = emb.cpu().numpy().astype(np.float16)
            b = emb_np.shape[0]
            feats[offset:offset+b] = emb_np
            for j in range(b):
                rows.append({
                    "idx": offset + j,
                    "instance_id": ids[j],
                    "ann_path": ann_paths[j],
                    "shape_idx": int(shape_idxs[j]),
                })
            offset += b

    feats.flush()
    pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8")
    print(f"[features] saved features -> {feat_path}")
    print(f"[features] saved meta -> {meta_path}")
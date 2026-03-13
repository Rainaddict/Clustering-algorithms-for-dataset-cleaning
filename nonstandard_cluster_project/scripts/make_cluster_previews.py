from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

from nscluster.image_crop import crop_instance_rgb
from nscluster.utils import ensure_dir


def make_montage(images, grid=10, thumb=96):
    # images: list[PIL]
    tiles = []
    for img in images:
        img = img.copy()
        img.thumbnail((thumb, thumb))
        # pad to thumb x thumb
        canvas = Image.new("RGB", (thumb, thumb), (0, 0, 0))
        x = (thumb - img.size[0]) // 2
        y = (thumb - img.size[1]) // 2
        canvas.paste(img, (x, y))
        tiles.append(canvas)

    W = grid * thumb
    H = grid * thumb
    out = Image.new("RGB", (W, H), (0, 0, 0))
    for i, tile in enumerate(tiles[:grid*grid]):
        r = i // grid
        c = i % grid
        out.paste(tile, (c * thumb, r * thumb))
    return out


def make_previews(cfg: dict):
    out_root = Path(cfg["out_root"])
    clusters_dir = out_root / "clusters"
    assign_path = clusters_dir / "assignments.csv"
    if not assign_path.exists():
        raise FileNotFoundError("assignments.csv not found. Run cluster step first.")

    previews_dir = clusters_dir / "previews"
    ensure_dir(previews_dir)

    preview_per_cluster = int(cfg["preview_per_cluster"])
    grid = int(cfg["preview_grid"])
    thumb = int(cfg["preview_thumb_size"])
    use_polygon_mask = bool(cfg["use_polygon_mask"])

    df = pd.read_csv(assign_path)
    # 优先拿 score 低的（更“典型”）或随机；这里用 score 升序取前 N
    html_lines = [
        "<html><head><meta charset='utf-8'><title>Cluster Previews</title></head><body>",
        "<h2>Cluster Previews</h2>",
        "<p>Open each image and decide cluster_id -> fine label mapping.</p>",
        "<ul>"
    ]

    for cid, g in tqdm(df.groupby("cluster_id"), desc="[preview] clusters"):
        g2 = g.sort_values("score", ascending=True).head(preview_per_cluster)
        imgs = []
        for _, row in g2.iterrows():
            # 需要 bbox_xyxy 与 polygon：assignments.csv里没有，只有 meta 信息
            # 所以这里从 index/instances.jsonl 再读取会更慢。
            # 为了不复杂化，本项目在 preview 阶段用 bbox 存在 assignments 里：
            # 你如果希望预览必须精确 mask，请在 index 阶段扩展 assignments。
            # 这里采取折中：从 ann 里重新读 shape polygon。
            from nscluster.ann_io import load_annotation, iter_shapes
            ann, _fmt = load_annotation(Path(row["ann_path"]))
            shapes = list(iter_shapes(ann))
            shape = shapes[int(row["shape_idx"])]
            pts = shape.get("points")
            # bbox 重新算（与 index 的 pad_ratio 会略有差异，但用于预览足够）
            from nscluster.geometry import polygon_to_bbox, expand_bbox, clamp_bbox
            bbox = polygon_to_bbox(pts)
            if bbox is None:
                continue
            bbox = expand_bbox(bbox, float(cfg["pad_ratio"]))
            img_w = int(ann.get("imageWidth", 0) or 0)
            img_h = int(ann.get("imageHeight", 0) or 0)
            if img_w > 0 and img_h > 0:
                bbox = clamp_bbox(bbox, img_w, img_h)

            # 定位图像
            from nscluster.ann_io import resolve_image_path
            rgb_root = Path(cfg["rgb_root"])
            # 简化：不再重建索引，优先 path/imageName；若失败会跳过
            img_path = resolve_image_path(ann, rgb_root, fname_map=None)

            if img_path is None:
                continue

            img = crop_instance_rgb(img_path, bbox, polygon=pts, use_polygon_mask=use_polygon_mask)
            imgs.append(img)

        if not imgs:
            continue

        montage = make_montage(imgs, grid=grid, thumb=thumb)
        out_img = previews_dir / f"cluster_{int(cid):06d}.jpg"
        montage.save(out_img, quality=90)

        html_lines.append(f"<li>cluster {cid}: <a href='previews/{out_img.name}'>{out_img.name}</a></li>")

    html_lines.append("</ul></body></html>")
    out_html = clusters_dir / "preview_index.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    print(f"[preview] saved previews -> {previews_dir}")
    print(f"[preview] index html -> {out_html}")
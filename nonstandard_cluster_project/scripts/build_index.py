from pathlib import Path
import json
from tqdm import tqdm

from nscluster.ann_io import load_annotation, iter_shapes, resolve_image_path
from nscluster.geometry import polygon_to_bbox, expand_bbox, clamp_bbox
from nscluster.utils import ensure_dir


def build_index(cfg: dict):
    label_root = Path(cfg["label_root"])
    rgb_root = Path(cfg["rgb_root"])
    out_root = Path(cfg["out_root"])

    target_label = cfg["target_label"]
    pad_ratio = float(cfg["pad_ratio"])
    min_box_size = int(cfg.get("min_box_size", 8))

    index_dir = out_root / "index"
    ensure_dir(index_dir)
    out_jsonl = index_dir / "instances.jsonl"
    out_bad = index_dir / "bad_records.jsonl"

    # 建立 filename -> fullpath 索引（兜底匹配 imageName）
    # 第一次 walk 可能稍慢，但只做一次，后续 resolve 可靠很多
    print(f"[index] building image filename index under: {rgb_root}")
    fname_map = {}
    for p in tqdm(rgb_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            fname_map[p.name] = str(p)

    ann_files = [p for p in label_root.rglob("*") if p.is_file()]
    print(f"[index] found annotation files: {len(ann_files)}")

    n_inst = 0
    n_bad = 0

    with open(out_jsonl, "w", encoding="utf-8") as fw, open(out_bad, "w", encoding="utf-8") as fb:
        for ann_path in tqdm(ann_files, desc="[index] scanning annotations"):
            try:
                ann, fmt = load_annotation(ann_path)
            except Exception as e:
                fb.write(json.dumps({"ann_path": str(ann_path), "error": f"parse_failed: {e}"}, ensure_ascii=False) + "\n")
                n_bad += 1
                continue

            img_path = resolve_image_path(ann, rgb_root, fname_map)
            if img_path is None:
                fb.write(json.dumps({"ann_path": str(ann_path), "error": "image_not_found"}, ensure_ascii=False) + "\n")
                n_bad += 1
                continue

            img_w = int(ann.get("imageWidth", 0) or 0)
            img_h = int(ann.get("imageHeight", 0) or 0)

            shapes = list(iter_shapes(ann))
            for shape_idx, shape in enumerate(shapes):
                main = shape.get("main", "")
                label = shape.get("label", "")
                if main != target_label and label != target_label:
                    continue

                pts = shape.get("points")
                if pts is None:
                    continue

                bbox = polygon_to_bbox(pts)
                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox
                if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                    continue

                bbox_pad = expand_bbox(bbox, pad_ratio)
                if img_w > 0 and img_h > 0:
                    bbox_pad = clamp_bbox(bbox_pad, img_w, img_h)

                instance_id = f"{ann_path.relative_to(label_root)}::shape_{shape_idx}"

                rec = {
                    "instance_id": instance_id,
                    "ann_path": str(ann_path),
                    "ann_format": fmt,
                    "shape_idx": shape_idx,
                    "image_path": img_path,
                    "image_width": img_w,
                    "image_height": img_h,
                    "bbox_xyxy": [float(v) for v in bbox_pad],
                    "polygon": pts,  # 原始 points（后续做 mask）
                }
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_inst += 1

    print(f"[index] done. instances={n_inst}, bad_files={n_bad}")
    print(f"[index] instances.jsonl -> {out_jsonl}")
    print(f"[index] bad_records.jsonl -> {out_bad}")
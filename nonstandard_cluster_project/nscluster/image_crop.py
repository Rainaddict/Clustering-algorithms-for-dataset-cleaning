from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

from nscluster.geometry import normalize_polygon


def crop_instance_rgb(image_path: str, bbox_xyxy, polygon=None, use_polygon_mask: bool = True):
    """
    返回 PIL RGB 图像（裁剪后的 crop）
    - bbox_xyxy: [x1,y1,x2,y2] float
    - polygon: 原始 points（用于 mask）
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 <= x1 or y2 <= y1:
        # 退化 bbox：返回 1x1
        return Image.fromarray(img[y1:y1+1, x1:x1+1])

    crop = img[y1:y2, x1:x2].copy()

    if use_polygon_mask and polygon is not None:
        pts = normalize_polygon(polygon)
        if len(pts) >= 3:
            pts_np = np.array([[p[0] - x1, p[1] - y1] for p in pts], dtype=np.int32)
            mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_np], 255)

            # 背景用 crop 的均值颜色（比纯黑更稳）
            bg = crop.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
            crop = (crop * (mask[..., None] / 255.0) + bg * (1.0 - mask[..., None] / 255.0)).astype(np.uint8)

    return Image.fromarray(crop)
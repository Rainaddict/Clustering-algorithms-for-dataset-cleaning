from __future__ import annotations
import math


def _normalize_points(points):
    """
    兼容 points 可能是：
    - [[x,y], [x,y], ...]
    - [{"0":x, "1":y}, ...]
    - {0:{0:x,1:y},1:{0:x,1:y},...}
    """
    if points is None:
        return None

    # list case
    if isinstance(points, list):
        out = []
        for p in points:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
            elif isinstance(p, dict):
                # dict with keys 0/1
                x = p.get(0, p.get("0"))
                y = p.get(1, p.get("1"))
                if x is not None and y is not None:
                    out.append((float(x), float(y)))
        return out if out else None

    # dict case: {idx: point}
    if isinstance(points, dict):
        out = []
        keys = sorted(points.keys(), key=lambda t: int(t) if str(t).isdigit() else str(t))
        for k in keys:
            p = points[k]
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
            elif isinstance(p, dict):
                x = p.get(0, p.get("0"))
                y = p.get(1, p.get("1"))
                if x is not None and y is not None:
                    out.append((float(x), float(y)))
        return out if out else None

    return None


def polygon_to_bbox(points):
    pts = _normalize_points(points)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    # 避免退化
    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return None
    return [x1, y1, x2, y2]


def expand_bbox(bbox_xyxy, pad_ratio: float):
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    px = w * pad_ratio
    py = h * pad_ratio
    return [x1 - px, y1 - py, x2 + px, y2 + py]


def clamp_bbox(bbox_xyxy, img_w: int, img_h: int):
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0.0, min(float(img_w - 1), x1))
    y1 = max(0.0, min(float(img_h - 1), y1))
    x2 = max(0.0, min(float(img_w - 1), x2))
    y2 = max(0.0, min(float(img_h - 1), y2))
    # 保证顺序
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_polygon(points):
    pts = _normalize_points(points)
    return pts if pts else []
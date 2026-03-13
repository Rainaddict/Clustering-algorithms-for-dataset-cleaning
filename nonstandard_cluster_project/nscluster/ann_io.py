from __future__ import annotations
from pathlib import Path
import json

try:
    import yaml
except Exception:
    yaml = None


def load_annotation(path: Path):
    s = path.read_text(encoding="utf-8", errors="ignore").strip()
    # 先尝试 JSON
    try:
        obj = json.loads(s)
        return obj, "json"
    except Exception:
        pass
    # 再尝试 YAML（YAML 可以 parse JSON 子集，很多“类 YAML”格式也能吃）
    if yaml is None:
        raise RuntimeError("pyyaml not installed. pip install pyyaml")
    try:
        obj = yaml.safe_load(s)
        if not isinstance(obj, dict):
            raise ValueError("annotation is not a dict")
        return obj, "yaml"
    except Exception as e:
        raise ValueError(f"Unsupported annotation format for {path}: {e}")


def dump_annotation(path: Path, ann: dict, fmt: str):
    if fmt == "json":
        path.write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # 默认 yaml
        if yaml is None:
            raise RuntimeError("pyyaml not installed. pip install pyyaml")
        path.write_text(yaml.safe_dump(ann, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _as_list_from_maybe_dict(x):
    # 兼容 shapes 是 list 或 dict(0,1,2...) 的情况
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        # 按数字 key 排序
        items = []
        for k in sorted(x.keys(), key=lambda t: int(t) if str(t).isdigit() else str(t)):
            items.append(x[k])
        return items
    return []


def iter_shapes(ann: dict):
    shapes = ann.get("shapes", [])
    shapes_list = _as_list_from_maybe_dict(shapes)
    for sh in shapes_list:
        if isinstance(sh, dict):
            yield sh


def resolve_image_path(ann: dict, rgb_root: Path, fname_map: dict | None):
    """
    优先：
    1) ann["path"] 作为相对 rgb_root 的路径
    2) ann["imageName"] 在 rgb_root 下按文件名索引
    3) ann["imageName"] 直接 join rgb_root
    """
    rel = ann.get("path")
    if isinstance(rel, str) and rel.strip():
        p = rgb_root / rel
        if p.exists():
            return str(p)

    name = ann.get("imageName")
    if isinstance(name, str) and name.strip():
        if fname_map is not None and name in fname_map:
            return fname_map[name]
        p2 = rgb_root / name
        if p2.exists():
            return str(p2)

    return None
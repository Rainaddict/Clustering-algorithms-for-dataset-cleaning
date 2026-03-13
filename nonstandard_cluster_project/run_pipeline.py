import argparse
from pathlib import Path
import yaml

from scripts.build_index import build_index
from scripts.extract_features import extract_features
from scripts.cluster_features import cluster_features
from scripts.make_cluster_previews import make_previews
from scripts.apply_cluster_labels import apply_cluster_labels


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config.yaml")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index")
    sub.add_parser("features")
    sub.add_parser("cluster")
    sub.add_parser("preview")

    ap = sub.add_parser("apply")
    ap.add_argument("--cluster-map", required=True, help="CSV: cluster_id,label")
    ap.add_argument("--overwrite-main", action="store_true",
                    help="Overwrite shape['main'] and shape['label'] to fine label. Default writes to shape['sub'].")

    args = parser.parse_args()
    cfg = load_config(args.config)

    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    if args.cmd == "index":
        build_index(cfg)
    elif args.cmd == "features":
        extract_features(cfg)
    elif args.cmd == "cluster":
        cluster_features(cfg)
    elif args.cmd == "preview":
        make_previews(cfg)
    elif args.cmd == "apply":
        apply_cluster_labels(cfg, args.cluster_map, args.overwrite_main)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
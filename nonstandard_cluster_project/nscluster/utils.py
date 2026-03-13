from __future__ import annotations
from pathlib import Path
import os
import random
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
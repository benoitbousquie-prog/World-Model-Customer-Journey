from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

def load_config(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

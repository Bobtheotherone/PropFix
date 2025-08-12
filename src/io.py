from __future__ import annotations
from pathlib import Path
import json, cv2, numpy as np, yaml

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return img

def save_image(path: str, img: np.ndarray) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)

def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def write_json(path: str, data: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

def load_manual_json(path: str|None) -> dict:
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    out={}
    for k in ["head_left","head_right","shoulder_l","shoulder_r","hip_l","hip_r",
              "elbow_l","elbow_r","wrist_l","wrist_r","wrists"]:
        if k in data:
            v=data[k]; out[k]=[(float(a),float(b)) for a,b in v] if k=="wrists" else (float(v[0]),float(v[1]))
    return out

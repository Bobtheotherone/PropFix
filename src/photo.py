from __future__ import annotations
from typing import Dict
import numpy as np, cv2
from .segments import Segment, SEGMENT_ORDER

def _to_hsv(img): return cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
def _from_hsv(hsv):
    hsv = np.clip(hsv, [0,0,0], [179,255,255]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _edit(img_bgr: np.ndarray, p: dict) -> np.ndarray:
    out = img_bgr.astype(np.float32)

    c = float(p.get("contrast", 1.0))
    b = float(p.get("brightness", 0.0))
    if c != 1.0 or b != 0.0:
        out = out * c + b

    sat = float(p.get("saturation", 1.0))
    hue = float(p.get("hue_deg", 0.0))
    if sat != 1.0 or hue != 0.0:
        hsv = _to_hsv(np.clip(out,0,255).astype(np.uint8))
        hsv[...,1] = np.clip(hsv[...,1] * sat, 0, 255)
        hsv[...,0] = (hsv[...,0] + (hue / 2.0)) % 180.0
        out = _from_hsv(hsv).astype(np.float32)

    blur = int(max(0, float(p.get("blur_px", 0))))
    if blur > 0:
        k = int(max(3, blur|1))
        out = cv2.GaussianBlur(out, (k,k), 0)

    sharp = float(p.get("sharpness", 1.0))
    if sharp != 1.0:
        base8 = np.clip(out, 0, 255).astype(np.uint8)
        ga = cv2.GaussianBlur(base8, (0,0), 1.0)
        usm = cv2.addWeighted(base8, sharp, ga, -(sharp-1.0), 0)
        out = usm.astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)

def apply_segment_edits(img_bgr: np.ndarray,
                        segments: Dict[str, Segment],
                        photo_controls: Dict[str, dict]) -> np.ndarray:
    """
    photo_controls[name] = {'brightness':-100..100,'contrast':0.5..1.5,
                            'saturation':0..2,'hue_deg':-30..30,
                            'blur_px':0..30,'sharpness':1..2}
    """
    base = img_bgr
    H, W = base.shape[:2]
    # Weighted average: start with base@weight=1
    acc = base.astype(np.float32)                           # (H,W,3)
    sumw = np.ones((H,W,1), np.float32)                     # start at 1 so base stays if no edits

    for name in SEGMENT_ORDER:
        if name not in photo_controls or name not in segments:
            continue
        params = photo_controls[name]
        if not params:
            continue

        seg = segments[name]
        edited = _edit(base, params).astype(np.float32)
        m = seg.mask.astype(np.float32)
        m = cv2.GaussianBlur(m, (11,11), 0)                 # soften edges
        m3 = m[...,None]                                    # (H,W,1)

        acc += m3 * edited
        sumw += m3

    out = (acc / np.maximum(sumw, 1e-6)).astype(np.uint8)
    return out

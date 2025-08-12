from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity("error")
except Exception:
    pass

from .io import load_image, save_image, read_yaml, load_manual_json
from .landmarks import detect_landmarks
from .segments import build_segments_from_landmarks, draw_annotation_overlay
from .warp_blend import blended_local_affine_warp
from .photo import apply_segment_edits
from .smart_lasso import (
    refine_segment_masks,
    build_lasso_vectors,
    make_masks_exclusive,
    render_lasso_overlay_png,
)

API_VERSION = "2-vectors"

def _safe_save_png(path: str, img: np.ndarray) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(p), img):
        raise RuntimeError(f"cv2.imwrite failed: {p}")

def _write_meta(meta_path: Path, payload: dict) -> None:
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass

def main() -> int:
    ap = argparse.ArgumentParser("Proportion toolkit with classifier + live warp")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--manual_json", default=None)
    ap.add_argument("--mode", choices=["classify", "warp"], default="classify")
    ap.add_argument("--controls_json", default=None)
    ap.add_argument("--meta_out", default=None, help="Optional sidecar JSON path for vectors/metadata")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    if args.input:  cfg["input"]  = args.input
    if args.output: cfg["output"] = args.output

    img_bgr = load_image(cfg["input"])
    pts = detect_landmarks(img_bgr)
    pts.update(load_manual_json(args.manual_json))

    if args.mode == "classify":
        segs = build_segments_from_landmarks(img_bgr.shape, pts)
        overlay = draw_annotation_overlay(img_bgr, segs, alpha=0.35, draw_edges=True)
        _safe_save_png(cfg["output"], overlay)
        meta = [{"name":k, "center":[float(v.center[0]), float(v.center[1])]} for k,v in segs.items()]
        payload = {"op":"classify","output":str(Path(cfg["output"]).resolve()),"segments":meta,"api_version":API_VERSION}
        if args.meta_out: _write_meta(Path(args.meta_out), payload)
        print(json.dumps(payload))
        return 0

    # --- warp mode ---
    segs = build_segments_from_landmarks(img_bgr.shape, pts)
    used_segments = list(segs.keys())

    controls = {}
    if args.controls_json and Path(args.controls_json).exists():
        with open(args.controls_json, "r", encoding="utf-8") as f:
            controls = json.load(f)

    try:
        refined = refine_segment_masks(img_bgr, segs, pts)
        if not refined:
            refined = {n: s.mask.astype(np.float32) for n, s in segs.items() if getattr(s, "mask", None) is not None}
        refined = make_masks_exclusive(refined, list(segs.keys()))
        for name, m in refined.items():
            segs[name].mask = m
    except Exception:
        refined = {n: s.mask.astype(np.float32) for n, s in segs.items() if getattr(s, "mask", None) is not None}

    warped = blended_local_affine_warp(img_bgr, segs, controls.get("geometry", {}), smooth_px=18)
    edited = apply_segment_edits(warped, segs, controls.get("photo", {}))
    _safe_save_png(cfg["output"], edited)

    vectors = build_lasso_vectors(refined, include_union_key=True)
    overlay_b64, overlay_data_url = render_lasso_overlay_png(img_bgr.shape, vectors)

    stats = {k: sum(len(poly) for poly in v) for k, v in vectors.items()}
    payload = {
        "op": "warp",
        "output": str(Path(cfg["output"]).resolve()),
        "used_segments": used_segments,
        "api_version": API_VERSION,
        "lasso_vectors": vectors,
        "lasso_overlay_png_b64": overlay_b64,
        "lasso_overlay_data_url": overlay_data_url,
        "vector_stats": {"segments": stats, "total_points": int(sum(stats.values()))}
    }
    if args.meta_out: _write_meta(Path(args.meta_out), payload)
    print(json.dumps(payload))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

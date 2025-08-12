# src/warp_rowuniform.py
from __future__ import annotations
import numpy as np
import cv2

EPS = 1e-6


def vertical_scales(pts, s_sh, s_hp, H, sigma_frac=0.30):
    sl, sr = pts["shoulder_l"], pts["shoulder_r"]
    hl, hr = pts["hip_l"], pts["hip_r"]

    shoulder_y = 0.5 * (sl[1] + sr[1])
    hip_y = 0.5 * (hl[1] + hr[1])

    ys = np.arange(H, dtype=np.float32)
    span = max(20.0, abs(hip_y - shoulder_y))
    sigma = max(1.0, sigma_frac * span)

    wS = np.exp(-0.5 * ((ys - shoulder_y) / sigma) ** 2)
    wH = np.exp(-0.5 * ((ys - hip_y) / sigma) ** 2)

    wsum = np.maximum(1.0, wS + wH)
    wS /= wsum
    wH /= wsum

    s_y = 1.0 + wS * (s_sh - 1.0) + wH * (s_hp - 1.0)
    return s_y.astype(np.float32), float(shoulder_y), float(hip_y)


def _vertical_feather_from_mask(mask01: np.ndarray, feather_px: int = 24) -> np.ndarray:
    """Cosine feather 0→1→0 across the vertical extent of the soft mask."""
    H, W = mask01.shape
    fg_rows = np.where(np.max(mask01, axis=1) > 1e-3)[0]
    v = np.zeros((H, 1), np.float32)
    if fg_rows.size == 0:
        return v
    y0, y1 = int(fg_rows[0]), int(fg_rows[-1])
    v[y0:y1 + 1, 0] = 1.0

    tband = min(feather_px, max(1, (y1 - y0 + 1) // 4))
    if tband > 0:
        tt = np.linspace(0, np.pi, tband, dtype=np.float32)
        v[y0:y0 + tband, 0] = 0.5 * (1.0 - np.cos(tt))
        v[y1 - tband + 1:y1 + 1, 0] = 0.5 * (1.0 - np.cos(tt[::-1]))
    return v


def _lateral_from_distance(mask01: np.ndarray) -> np.ndarray:
    """Raised-cosine from edge (0) to row-center (1) using distance transform."""
    inside = (mask01 > 1e-3).astype(np.uint8)
    if inside.max() == 0:
        return np.zeros_like(mask01, np.float32)

    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    H, W = inside.shape
    w = np.zeros((H, W), np.float32)
    for y in range(H):
        row = dist[y, :]
        m = row.max()
        if m > 0:
            t = np.clip(row / m, 0.0, 1.0)
            w[y, :] = 0.5 * (1.0 - np.cos(np.pi * t))
    return w


def influence_from_mask(torso01: np.ndarray,
                        v_feather_px: int = 24,
                        blur_ksize: int = 15) -> np.ndarray:
    """
    Build a smooth 2D influence map in [0,1] from a soft torso mask:
      - lateral falloff via distance transform (continuous in x and y)
      - vertical cosine feather at mask top/bottom
      - overall Gaussian blur to remove micro-steps
    """
    lat = _lateral_from_distance(torso01)
    vgate = _vertical_feather_from_mask(torso01, feather_px=v_feather_px)
    w = lat * vgate

    k = max(3, blur_ksize | 1)  # odd
    w = cv2.GaussianBlur(w, (k, k), 0)
    return np.clip(w, 0.0, 1.0).astype(np.float32)


def build_maps(shape, pts, torso01, s_sh, s_hp,
               sigma_frac=0.30, s_floor=0.85,
               v_feather_px: int = 24, blur_ksize: int = 15):
    """
    Row-uniform warp:
      x_in = x_mid + (x_out - x_mid) / s_eff(y,x)
      s_eff(y,x) = 1 + influence(y,x) * (s_row(y) - 1)
    """
    H, W = shape[:2]

    # per-row target scales
    s_y, sy, hy = vertical_scales(pts, s_sh, s_hp, H, sigma_frac)
    s_row = np.maximum(s_y[:, None], s_floor)

    # torso influence (soft 2D)
    influence = influence_from_mask(torso01, v_feather_px=v_feather_px, blur_ksize=blur_ksize)

    # midline x(y)
    sl, sr = pts["shoulder_l"], pts["shoulder_r"]
    hl, hr = pts["hip_l"], pts["hip_r"]
    mid_sx = 0.5 * (sl[0] + sr[0])
    mid_hx = 0.5 * (hl[0] + hr[0])

    ys = np.arange(H, dtype=np.float32)
    if abs(hy - sy) < 1.0:
        x_mid_row = np.full((H,), float(mid_sx), np.float32)
    else:
        t = (ys - sy) / (hy - sy)
        x_mid_row = ((1.0 - t) * mid_sx + t * mid_hx).astype(np.float32)
    x_mid = x_mid_row[:, None]

    # effective scale field
    s_eff = 1.0 + influence * (s_row - 1.0)
    s_eff = cv2.GaussianBlur(s_eff, (1, 7), 0)  # vertical smoothing

    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_x = x_mid + (X - x_mid) / np.maximum(s_eff, EPS)
    map_y = Y
    return map_x.astype(np.float32), map_y.astype(np.float32), x_mid_row, s_y, influence


def remap_img(img, map_x, map_y):
    # Lanczos minimizes ringing/striping on fabrics vs bicubic
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REFLECT101)


def remap_mask(mask01, map_x, map_y):
    # Keep soft float mask across passes (no threshold)
    return cv2.remap(mask01, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)


def pad_for_warp(img, pad_px):
    return cv2.copyMakeBorder(img, 0, 0, pad_px, pad_px, cv2.BORDER_REFLECT101)

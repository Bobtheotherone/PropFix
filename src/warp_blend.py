# src/warp_blend.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import cv2

# If segments module is available, import metadata/types; otherwise keep loose typing.
try:
    from .segments import Segment, SEGMENT_ORDER
except Exception:
    Segment = object  # type: ignore
    SEGMENT_ORDER: List[str] = []

EPS = 1e-6


# ------------------------- utilities -------------------------

def _ensure_f32_mask(m: np.ndarray | None) -> np.ndarray | None:
    """Return a float32 mask in [0,1] or None."""
    if m is None:
        return None
    m = m.astype(np.float32, copy=False)
    if m.max() > 1.001:
        m *= (1.0 / 255.0)
    return np.clip(m, 0.0, 1.0)


def _build_support_and_norm(ws: np.ndarray,
                            support_thr: float = 0.03,
                            soften_hi: float = 0.12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize N masks so they sum to 1 where total weight > support_thr.
    Returns:
      ws_norm: (N,H,W) normalized weights (zeros outside support)
      support_mask: (H,W) bool where sum(ws) > support_thr
      influence: (H,W) float in [0,1], soft ramp from support_thr..soften_hi
    """
    s = ws.sum(axis=0)  # (H,W)
    support = s > support_thr
    out = np.zeros_like(ws, dtype=np.float32)
    if np.any(support):
        out[:, support] = ws[:, support] / s[None, support]
    influence = np.clip((s - support_thr) / max(soften_hi - support_thr, 1e-6), 0.0, 1.0)
    return out, support, influence


def _R(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s,  c]], np.float32)


def _affine_from_axes(ex_vec, ey_vec, sx, sy, theta_deg):
    """
    Forward local->global linear part T: scale by (sx,sy) along local axes,
    then rotate by theta_deg; expressed in global coordinates.
    """
    ex = np.array(ex_vec, np.float32)
    ey = np.array(ey_vec, np.float32)
    exu = ex / (np.linalg.norm(ex) + EPS)
    eyu = ey / (np.linalg.norm(ey) + EPS)
    Rw = np.stack([exu, eyu], axis=1).astype(np.float32)  # columns are local axes in global coords
    S  = np.diag([sx, sy]).astype(np.float32)
    Rl = _R(np.deg2rad(theta_deg)).astype(np.float32)
    try:
        Rw_inv = np.linalg.inv(Rw)
    except np.linalg.LinAlgError:
        Rw_inv = np.linalg.pinv(Rw)
    T = Rw @ (Rl @ S) @ Rw_inv
    return T


def _synthesize_gaussian_mask(H, W, center_xy, ex_vec, ey_vec, k_scale=0.8) -> np.ndarray:
    """Fallback soft mask aligned to local axes."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ex = np.array(ex_vec, np.float32)
    ey = np.array(ey_vec, np.float32)

    sigma_floor = max(0.06 * min(H, W), 30.0)
    sigx = max(sigma_floor, float(np.linalg.norm(ex)) * k_scale)
    sigy = max(sigma_floor, float(np.linalg.norm(ey)) * k_scale)

    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij"
    )
    Xc = xx - cx
    Yc = yy - cy

    exu = ex / (np.linalg.norm(ex) + EPS)
    eyu = ey / (np.linalg.norm(ey) + EPS)
    u = Xc * exu[0] + Yc * exu[1]
    v = Xc * eyu[0] + Yc * eyu[1]

    m = np.exp(-0.5 * ((u / sigx) ** 2 + (v / sigy) ** 2)).astype(np.float32)
    m -= m.min()
    mx = m.max()
    if mx > EPS:
        m /= mx
    return m


def _min_singular_value_2x2(a11, a12, a21, a22):
    """sigma_min of 2x2 field A via sqrt(min eig of A^T A)."""
    s11 = a11 * a11 + a21 * a21
    s12 = a11 * a12 + a21 * a22
    s22 = a12 * a12 + a22 * a22
    tr = s11 + s22
    det = s11 * s22 - s12 * s12
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    lam_min = 0.5 * (tr - np.sqrt(disc))
    lam_min = np.maximum(lam_min, EPS)
    return np.sqrt(lam_min)


def _build_mipmap(img_u8: np.ndarray) -> List[np.ndarray]:
    """Gaussian pyramid [L0(full), L1(1/2), ...] until min(H,W)<16. Input must be uint8."""
    levels = [img_u8]
    H, W = img_u8.shape[:2]
    while min(H, W) >= 16:
        levels.append(cv2.pyrDown(levels[-1]))
        H, W = levels[-1].shape[:2]
    return levels


# ------------------------- main warp -------------------------

def blended_local_affine_warp(img_bgr: np.ndarray,
                              segments: Dict[str, "Segment"],
                              geometry: Dict[str, Dict[str, float]],
                              smooth_px: int = 18) -> np.ndarray:
    """
    Backward (dest->source) warp with:
      • Per-part anisotropic linear maps blended by normalized supports.
      • Mild Jacobian tempering to avoid foldovers.
      • Per-pixel mip LOD sampling.
      • Vacated-only inpainting + premultiplied sprite compositing.
    """
    H, W = img_bgr.shape[:2]

    # Destination pixel grid
    xx, yy = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="xy"
    )

    # Segment order
    seg_keys = list(segments.keys())
    if SEGMENT_ORDER:
        ordered = [n for n in SEGMENT_ORDER if n in segments]
        names = ordered if ordered else seg_keys
    else:
        names = seg_keys
    if not names:
        return img_bgr.copy()

    # Prepare masks (raw or synthesized)
    raw_masks: List[np.ndarray | None] = []
    total_sum = 0.0
    for n in names:
        m = _ensure_f32_mask(getattr(segments[n], "mask", None))
        raw_masks.append(m)
        if m is not None:
            total_sum += float(m.sum())
    synth_all = (total_sum < 0.005 * H * W)

    masks:   List[np.ndarray] = []
    centers: List[np.ndarray] = []

    # inverse linear parts (for backward mapping)
    B00: List[float] = []
    B01: List[float] = []
    B10: List[float] = []
    B11: List[float] = []
    t0:  List[float] = []
    t1:  List[float] = []

    for i, name in enumerate(names):
        seg = segments[name]
        g = geometry.get(name, {}) if geometry is not None else {}

        # conservative clamps
        sx  = float(np.clip(g.get("sx", 1.0), 0.5, 1.6))
        sy  = float(np.clip(g.get("sy", 1.0), 0.5, 1.6))
        rot = float(np.clip(g.get("rot_deg", 0.0), -35.0, 35.0))
        tx  = float(g.get("tx", 0.0))
        ty  = float(g.get("ty", 0.0))

        ex = getattr(seg, "ex", (1.0, 0.0))
        ey = getattr(seg, "ey", (0.0, 1.0))

        # forward then invert for backward mapping
        T = _affine_from_axes(ex, ey, sx, sy, rot)
        try:
            Binv = np.linalg.inv(T).astype(np.float32)
        except np.linalg.LinAlgError:
            Binv = np.linalg.pinv(T).astype(np.float32)
        B00.append(float(Binv[0, 0])); B01.append(float(Binv[0, 1]))
        B10.append(float(Binv[1, 0])); B11.append(float(Binv[1, 1]))

        # local-axis translation expressed in global coords
        t = (np.array(ex, np.float32).reshape(2) * tx +
             np.array(ey, np.float32).reshape(2) * ty)
        t0.append(float(t[0]))
        t1.append(float(t[1]))

        centers.append(np.array([float(seg.center[0]), float(seg.center[1])],
                                np.float32))

        # mask (raw or synthesized) + feather
        m = raw_masks[i]
        weak = (m is None) or (float(m.sum()) < 0.005 * H * W)
        if synth_all or weak:
            m = _synthesize_gaussian_mask(H, W, seg.center, ex, ey, k_scale=0.65)
        if smooth_px and smooth_px > 0:
            k = int(max(3, (int(smooth_px) | 1)))
            try:
                m = cv2.GaussianBlur(m, (k, k), 0)
            except Exception:
                m = cv2.GaussianBlur(m, (0, 0), 1.0)
        masks.append(np.clip(m.astype(np.float32), 0.0, 1.0))

    # Normalize only inside compact support; get soft influence ramp
    ws_raw = np.stack(masks, axis=0)                       # (N,H,W)
    ws, support_mask, influence = _build_support_and_norm(ws_raw,
                                                          support_thr=0.03,
                                                          soften_hi=0.12)
    if not np.any(support_mask):
        return img_bgr.copy()

    # Approximate blended inverse Jacobian inside support
    Bxx = np.zeros((H, W), np.float32)
    Bxy = np.zeros((H, W), np.float32)
    Byx = np.zeros((H, W), np.float32)
    Byy = np.zeros((H, W), np.float32)
    for i in range(len(names)):
        w = ws[i]
        Bxx += w * B00[i]
        Bxy += w * B01[i]
        Byx += w * B10[i]
        Byy += w * B11[i]

    sigma_min = _min_singular_value_2x2(Bxx, Bxy, Byx, Byy)

    # Mild tempering only where we have support and compression is high
    s_hi, s_lo = 0.70, 0.55
    temper = np.ones_like(sigma_min, dtype=np.float32)
    temper[~support_mask] = 1.0
    mask_lo = support_mask & (sigma_min < s_lo)
    mask_hi = support_mask & (sigma_min >= s_lo) & (sigma_min < s_hi)
    temper[mask_hi] = ((sigma_min[mask_hi] - s_lo) / (s_hi - s_lo))**2
    temper[mask_lo] = 0.0
    # ease with influence ramp
    temper = 1.0 - influence * (1.0 - temper)

    # -------- Blended backward displacement field (correct translation term) -----
    delta_x = np.zeros((H, W), np.float32)
    delta_y = np.zeros((H, W), np.float32)

    for i, _name in enumerate(names):
        cx, cy = centers[i]
        rel_x = xx - cx
        rel_y = yy - cy

        # Correct formula: d = (B - I)*(y - c) - t
        bx = B00[i] * rel_x + B01[i] * rel_y
        by = B10[i] * rel_x + B11[i] * rel_y
        dx = (bx - rel_x) - t0[i]
        dy = (by - rel_y) - t1[i]

        w = ws[i]
        delta_x += w * dx
        delta_y += w * dy

    # Apply tempering only inside influenced regions
    delta_x *= temper
    delta_y *= temper
    delta_x[~support_mask] = 0.0
    delta_y[~support_mask] = 0.0

    # Final backward sampling grid
    map_x = xx + delta_x
    map_y = yy + delta_y
    np.clip(map_x, 0.0, float(W - 1), out=map_x)
    np.clip(map_y, 0.0, float(H - 1), out=map_y)

    # ======================= Vacated-only compositing =======================

    # 1) Full person mask as union of part masks
    person_mask = np.zeros((H, W), np.float32)
    for m in masks:
        person_mask = np.maximum(person_mask, m)
    person_mask = np.clip(person_mask, 0.0, 1.0)

    # 2) Destination coverage of the warped person (same backward map)
    warped_person = cv2.remap(
        person_mask, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    warped_person = np.clip(warped_person, 0.0, 1.0)

    # 3) Vacated background = covered before, not covered after
    vacated = np.clip(person_mask - warped_person, 0.0, 1.0)

    # Clean the inpaint mask (remove specks; avoid over-inpainting)
    ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ker5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vacated = cv2.morphologyEx(vacated, cv2.MORPH_OPEN, ker3, iterations=1)
    vacated = cv2.morphologyEx(vacated, cv2.MORPH_DILATE, ker3, iterations=1)
    vacated_u8 = np.clip(vacated * 255.0, 0, 255).astype(np.uint8)

    # 4) Inpaint ONLY the vacated background on a copy of the original image
    try:
        dest_bg = cv2.inpaint(
            img_bgr, (vacated_u8 > 0).astype(np.uint8) * 255,
            3, cv2.INPAINT_TELEA
        )
    except Exception:
        dest_bg = img_bgr.copy()

    # 5) Sprite = premultiplied person layer (RGB * alpha). Regions outside person are zero.
    alpha = person_mask[..., None].astype(np.float32)  # (H,W,1)
    sprite_premul = img_bgr.astype(np.float32) * alpha

    # 6) Build pyramids for per-pixel LOD sampling (uint8)
    mip_rgb = _build_mipmap(np.clip(sprite_premul, 0, 255).astype(np.uint8))
    mip_a   = _build_mipmap((alpha * 255.0).astype(np.uint8))
    Lr = len(mip_rgb) - 1

    # LOD map: anti-shrink blur limited to support
    sigma_c = np.clip(sigma_min, 1e-6, 8.0)
    lod_raw = np.maximum(0.0, np.log2(1.0 / np.minimum(1.0, sigma_c)))
    lambda_map = influence * lod_raw
    lambda_map[~support_mask] = 0.0

    i0 = np.floor(lambda_map).astype(np.int32)
    i1 = np.minimum(i0 + 1, Lr).astype(np.int32)
    t  = (lambda_map - i0).astype(np.float32)

    warped_premul = np.zeros((H, W, 3), np.float32)
    warped_alpha  = np.zeros((H, W, 1), np.float32)

    # 7) LOD-aware backward sampling with constant-zero borders (safe for premultiplied)
    for lvl in range(Lr + 1):
        scale = float(1 << lvl)

        # RGB (premultiplied)
        img_lvl = mip_rgb[lvl].astype(np.float32)
        Hl, Wl = img_lvl.shape[:2]
        mx = map_x / scale
        my = map_y / scale
        np.clip(mx, 0.0, float(Wl - 1), out=mx)
        np.clip(my, 0.0, float(Hl - 1), out=my)
        samp_rgb = cv2.remap(
            img_lvl, mx, my,
            interpolation=(cv2.INTER_LINEAR if lvl else cv2.INTER_LANCZOS4),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        # Alpha
        a_lvl = mip_a[lvl].astype(np.float32)
        Hl, Wl = a_lvl.shape[:2]
        ma = map_x / scale
        na = map_y / scale
        np.clip(ma, 0.0, float(Wl - 1), out=ma)
        np.clip(na, 0.0, float(Hl - 1), out=na)
        samp_a = cv2.remap(
            a_lvl, ma, na,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        w = ((i0 == lvl).astype(np.float32) * (1.0 - t) +
             (i1 == lvl).astype(np.float32) * t)[..., None]
        warped_premul += samp_rgb * w
        warped_alpha  += (samp_a[..., None]) * w

    # 8) Un-premultiply safely; clamp alpha to avoid halos
    warped_alpha = np.clip(warped_alpha / 255.0, 0.0, 1.0).astype(np.float32)
    denom = np.maximum(warped_alpha, 0.03)  # alpha floor
    warped_rgb = np.clip(warped_premul / denom, 0.0, 255.0).astype(np.uint8)

    # 9) Tight paste mask to avoid cloning low-alpha fringes
    warped_mask_u8 = np.clip(warped_alpha[..., 0] * 255.0, 0, 255).astype(np.uint8)
    warped_mask_u8 = cv2.morphologyEx(warped_mask_u8, cv2.MORPH_ERODE, ker3, iterations=1)
    warped_mask_u8 = cv2.morphologyEx(warped_mask_u8, cv2.MORPH_DILATE, ker5, iterations=1)

    # 10) Clone center from mask moments; fallback to center
    M = cv2.moments(warped_mask_u8, binaryImage=True)
    if M["m00"] > 1.0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = W // 2, H // 2

    # 11) Final composite
    try:
        out = cv2.seamlessClone(
            warped_rgb, dest_bg, warped_mask_u8, (cx, cy), cv2.NORMAL_CLONE
        )
    except Exception:
        m3 = (warped_mask_u8.astype(np.float32) / 255.0)[..., None]
        out = (warped_rgb.astype(np.float32) * m3 +
               dest_bg.astype(np.float32) * (1.0 - m3)).astype(np.uint8)

    return out

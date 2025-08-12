from __future__ import annotations
from typing import Dict, Tuple, Iterable, List
import numpy as np
import cv2
import base64

# Project types
try:
    from .segments import Segment, SEGMENT_ORDER
except Exception:
    Segment = object  # type: ignore
    SEGMENT_ORDER = []

try:
    from .seg import build_person_mask
except Exception:
    # Safety fallback
    def build_person_mask(img_bgr, pts):  # type: ignore
        h, w = img_bgr.shape[:2]
        return np.ones((h, w), np.float32)

GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD = 0, 1, 2, 3

HEAD_AUX_NAMES: Tuple[str, ...] = (
    "hair", "headband", "ribbon", "ribbon_tail", "bandana", "band", "head_accessory"
)

# -------------------------- utilities --------------------------

def _bbox_from_mask(mask01: np.ndarray, thr: float = 0.20) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask01 > thr)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def _pad_rect(x0: int, y0: int, x1: int, y1: int, pad: int, W: int, H: int) -> Tuple[int, int, int, int]:
    return max(0, x0 - pad), max(0, y0 - pad), min(W - 1, x1 + pad), min(H - 1, y1 + pad)

def _soft(mask01: np.ndarray, k: int) -> np.ndarray:
    k = int(max(3, (k | 1)))
    return cv2.GaussianBlur(mask01.astype(np.float32), (k, k), 0)

def _morph_kernel(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

def _largest_components_touching_or_red(
    refined01: np.ndarray,
    prior_bin: np.ndarray,
    red_keep: np.ndarray | None,
    keep_k: int,
    dilate_r: int,
) -> np.ndarray:
    if refined01.max() <= 0:
        return refined01
    prior_dil = cv2.dilate(prior_bin.astype(np.uint8), _morph_kernel(max(1, dilate_r)), 1)
    rbin = (refined01 > 0.5).astype(np.uint8)
    num, labels = cv2.connectedComponents(rbin, connectivity=8)
    if num <= 1:
        return refined01
    areas = []
    for i in range(1, num):
        comp = (labels == i)
        touch = (comp & (prior_dil > 0)).any()
        redok = False
        if (not touch) and red_keep is not None:
            redok = (comp & (red_keep > 0)).any()
        if not (touch or redok):
            continue
        areas.append((int(comp.sum()), i))
    if not areas:
        return np.zeros_like(refined01, dtype=np.float32)
    areas.sort(reverse=True)
    keep_ids = {i for _, i in areas[:keep_k]}
    kept = np.zeros_like(rbin)
    for i in keep_ids:
        kept |= (labels == i)
    return refined01 * kept.astype(np.float32)

def _distance_gate(refined01: np.ndarray, prior_bin: np.ndarray, tau_px: float) -> np.ndarray:
    if refined01.max() <= 0:
        return refined01
    support = (prior_bin > 0).astype(np.uint8)
    if support.max() == 0:
        return refined01
    inv = (1 - support).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    if tau_px <= 1:
        return refined01
    gate = np.exp(-(dist / float(tau_px)) ** 2)
    return refined01 * gate.astype(np.float32)

# --------------------- color accessory detection ---------------------

def _red_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = ((h < 14) | (h > 170)) & (s > 70) & (v > 40)
    red = red.astype(np.uint8) * 255
    red = cv2.medianBlur(red, 5)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, _morph_kernel(2))
    red = cv2.dilate(red, _morph_kernel(1), 1)
    return (red > 0).astype(np.uint8)

def _detect_red_accessories_near_head(img_bgr: np.ndarray, head_bb: Tuple[int, int, int, int]) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    x0, y0, x1, y1 = head_bb
    cw = max(8, x1 - x0 + 1)
    cx = (x0 + x1) // 2
    red = _red_mask(img_bgr)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(red, 8)
    if num <= 1:
        return np.zeros((H, W), np.float32)
    band_top = y0 - int(0.8 * (y1 - y0))
    band_bot = y1 + int(0.8 * (y1 - y0))
    out = np.zeros_like(red, np.uint8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        rcx, rcy = cents[i]
        if area < 120: continue
        if rcy < band_top or rcy > band_bot: continue
        if abs(rcx - cx) > 2.0 * cw: continue
        out[labels == i] = 1
    return _soft(out.astype(np.float32), 7)

# --------------------- seeding & refinement ---------------------

def _to_gc_roi(H: int, W: int, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    gc = np.full((H, W), GC_BGD, np.uint8)
    gc[y0:y1 + 1, x0:x1 + 1] = GC_PR_BGD
    return gc

def _seed_confidence(
    name: str,
    seg01_for_seed: np.ndarray,
    person01: np.ndarray,
    img_bgr: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    H, W = seg01_for_seed.shape
    x0, y0, x1, y1 = roi
    seg_bin = (seg01_for_seed > 0.4).astype(np.uint8)
    roi_w, roi_h = (x1 - x0 + 1), (y1 - y0 + 1)
    base_r = max(1, int(round(0.06 * min(roi_w, roi_h))))
    ker_in = _morph_kernel(base_r)
    ker_near = _morph_kernel(max(1, int(1.8 * base_r)))
    sure_fg = cv2.erode(seg_bin, ker_in, 1)
    near_field = cv2.dilate(seg_bin, ker_near, 1)
    sp = person01[y0:y1 + 1, x0:x1 + 1]
    nb_person = (person01 < 0.25).astype(np.uint8)
    headish = name in ("head", "hair", "ribbon", "headband", "ribbon_tail", "bandana")
    gc = _to_gc_roi(H, W, roi)
    view = gc[y0:y1 + 1, x0:x1 + 1]
    sf = sure_fg[y0:y1 + 1, x0:x1 + 1]
    view[sf > 0] = GC_FGD
    near_big = (near_field[y0:y1 + 1, x0:x1 + 1] > 0)
    nb = nb_person[y0:y1 + 1, x0:x1 + 1]
    far_outside = (nb > 0) & (~near_big)
    if not headish:
        view[far_outside] = GC_BGD
    else:
        view[far_outside] = np.where(view[far_outside] == GC_FGD, GC_FGD, GC_PR_BGD)
    pr_fg = ((sp > 0.25) | near_big) & (sf == 0)
    view[pr_fg] = np.where(view[pr_fg] == GC_FGD, GC_FGD, GC_PR_FGD)
    gray = cv2.cvtColor(img_bgr[y0:y1 + 1, x0:x1 + 1, :], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_buf = cv2.dilate(edges, _morph_kernel(max(1, base_r // 2)))
    view[(edge_buf > 0) & (sf == 0)] = np.where(
        view[(edge_buf > 0) & (sf == 0)] == GC_FGD, GC_FGD, GC_PR_FGD
    )
    return gc

def _grabcut_refine(img_bgr: np.ndarray, gc_mask: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    roi_img = img_bgr[y0:y1 + 1, x0:x1 + 1, :]
    roi_gc = gc_mask[y0:y1 + 1, x0:x1 + 1].copy()
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(roi_img, roi_gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    m_roi = ((roi_gc == GC_FGD) | (roi_gc == GC_PR_FGD)).astype(np.float32)
    m_full = np.zeros(img_bgr.shape[:2], np.float32)
    m_full[y0:y1 + 1, x0:x1 + 1] = m_roi
    feather = max(5, int(round(0.02 * (max(1, (x1 - x0) + (y1 - y0))))))
    m_full = _soft(m_full, feather)
    return np.clip(m_full, 0.0, 1.0)

# --------------------- refinement driver ---------------------

def refine_segment_masks(
    img_bgr: np.ndarray,
    segments: Dict[str, Segment],
    pts: Dict[str, Tuple[float, float]] | None = None,
) -> Dict[str, np.ndarray]:
    H, W = img_bgr.shape[:2]
    out: Dict[str, np.ndarray] = {}
    try:
        person01 = build_person_mask(img_bgr, pts or {})
    except Exception:
        person01 = np.ones((H, W), np.float32)

    raw: Dict[str, np.ndarray] = {}
    for name in (SEGMENT_ORDER or list(segments.keys())):
        seg = segments.get(name)
        if seg is not None and getattr(seg, "mask", None) is not None:
            raw[name] = seg.mask.astype(np.float32)

    neck_prior = raw.get("neck", None)

    for name in (SEGMENT_ORDER or list(segments.keys())):
        seg = segments.get(name)
        if seg is None or getattr(seg, "mask", None) is None:
            continue
        try:
            seg01_prior = raw[name]
            headish = name in ("head", "hair", "ribbon", "headband", "ribbon_tail", "bandana")
            red_keep = None
            if name == "head":
                extras = [raw[n] for n in HEAD_AUX_NAMES if n in raw]
                seed = seg01_prior
                if extras:
                    seed = np.maximum(seed, np.max(np.stack(extras, axis=0), axis=0))
                bb = _bbox_from_mask(seg01_prior, thr=0.20)
                if bb is not None:
                    red_keep = _detect_red_accessories_near_head(img_bgr, bb)
                    if red_keep is not None:
                        seed = np.maximum(seed, red_keep)
                seg01_for_seed = seed
            else:
                seg01_for_seed = seg01_prior

            bb = _bbox_from_mask(seg01_for_seed, thr=0.20)
            if bb is None:
                out[name] = seg01_prior
                continue
            pad_scale = 0.12 if headish else 0.06
            pad = max(6, int(round(pad_scale * min(W, H))))
            x0, y0, x1, y1 = _pad_rect(*bb, pad, W, H)
            gc_mask = _seed_confidence(name, seg01_for_seed, person01, img_bgr, (x0, y0, x1, y1))
            refined = _grabcut_refine(img_bgr, gc_mask, (x0, y0, x1, y1))
            refined = _largest_components_touching_or_red(
                refined, (seg01_for_seed > 0.4),
                red_keep if name == "head" else None,
                keep_k=5, dilate_r=max(2, pad // 3)
            )
            tau = max(5, pad * (0.6 if headish else 0.9))
            refined = _distance_gate(refined, (seg01_for_seed > 0.4), tau_px=tau)
            if name == "head" and neck_prior is not None:
                neck_dil = cv2.dilate((neck_prior > 0.4).astype(np.uint8), _morph_kernel(max(2, pad // 3)))
                refined *= (1.0 - neck_dil.astype(np.float32))
            if headish:
                refined *= np.clip(person01 + 0.20, 0.0, 1.0)
                refined = np.maximum(refined, _soft(seg01_prior, 5))
            else:
                refined *= np.clip(person01 * 1.25, 0.0, 1.0)
                refined = np.maximum(refined, _soft(seg01_prior, 5))
            out[name] = np.clip(refined, 0.0, 1.0).astype(np.float32)
        except Exception:
            out[name] = seg.mask.astype(np.float32)
    return out

# --------------------- exclusivity ---------------------

def make_masks_exclusive(refined: Dict[str, np.ndarray], priority: Iterable[str]) -> Dict[str, np.ndarray]:
    names = [n for n in priority if n in refined] or list(refined.keys())
    H, W = next(iter(refined.values())).shape
    assigned = np.zeros((H, W), np.float32)
    out: Dict[str, np.ndarray] = {}
    for n in names:
        m = np.clip(refined[n], 0.0, 1.0)
        m = m * (1.0 - assigned)
        if m.max() > 0:
            m = _soft(m, 3)
        out[n] = np.clip(m, 0.0, 1.0)
        assigned = np.maximum(assigned, out[n])
    for n, m in refined.items():
        if n in out:
            continue
        mm = np.clip(m, 0.0, 1.0) * (1.0 - assigned)
        out[n] = mm
        assigned = np.maximum(assigned, mm)
    return out

# --------------------- vectors & overlay ---------------------

def _contours_from_mask(mask01: np.ndarray, min_area: float = 64.0, approx_eps: float = 2.0) -> List[List[Tuple[int,int]]]:
    binm = (mask01 > 0.5).astype(np.uint8) * 255
    if binm.max() == 0:
        return []
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[Tuple[int,int]]] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        peri = float(cv2.arcLength(c, True))
        eps = max(approx_eps, 0.01 * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        polys.append(pts)
    return polys

def build_lasso_vectors(refined: Dict[str, np.ndarray], include_union_key: bool = True) -> Dict[str, List[List[Tuple[int,int]]]]:
    out: Dict[str, List[List[Tuple[int,int]]]] = {}
    union = None
    for name, m in refined.items():
        out[name] = _contours_from_mask(m, min_area=64.0, approx_eps=2.0)
        union = m if union is None else np.maximum(union, m)
    if include_union_key and union is not None:
        out["person"] = _contours_from_mask(union, min_area=128.0, approx_eps=2.0)
    return out

_PALETTE_HEX = {
    "head":"#3f80ff", "neck":"#ffd85a", "torso":"#ff50ff",
    "upper_arm_l":"#50c850", "upper_arm_r":"#50c8c8",
    "forearm_l":"#54b854", "forearm_r":"#54b8c8",
    "hand_l":"#3cb37c",  "hand_r":"#3cb3c3",
    "thigh_l":"#b478ff", "thigh_r":"#ff7878",
    "shin_l":"#a46cff",  "shin_r":"#ff6c6c",
    "foot_l":"#9670ff",  "foot_r":"#ff7070",
    "hair":"#aaaaaa", "headband":"#4040ff", "ribbon":"#4040ff", "ribbon_tail":"#4040ff",
    "person":"#ffffff"
}

def _hex_to_bgr(hex_str: str) -> Tuple[int,int,int]:
    h = hex_str.lstrip("#")
    if len(h) == 3:
        r = int(h[0]*2,16); g = int(h[1]*2,16); b = int(h[2]*2,16)
    else:
        r = int(h[0:2],16); g = int(h[2:4],16); b = int(h[4:6],16)
    return (b,g,r)

def _draw_dashed_polyline(img_bgra: np.ndarray, pts: List[Tuple[int,int]], color_bgra: Tuple[int,int,int,int], dash_px: int = 8, gap_px: int = 8, thickness: int = 2):
    def _lerp(p0, p1, t):
        return (p0[0] + (p1[0]-p0[0])*t, p0[1] + (p1[1]-p0[1])*t)
    n = len(pts)
    if n < 2: return
    segs = []
    for i in range(n):
        p0 = pts[i]; p1 = pts[(i+1)%n]
        dx = p1[0]-p0[0]; dy = p1[1]-p0[1]
        L = (dx*dx + dy*dy) ** 0.5
        if L < 1: continue
        t = 0.0; on = True
        while t < 1.0:
            step = (dash_px if on else gap_px) / L
            t2 = min(1.0, t + step)
            if on:
                a = _lerp(p0,p1,t); b = _lerp(p0,p1,t2)
                segs.append(((int(round(a[0])),int(round(a[1]))),(int(round(b[0])),int(round(b[1])))))
            on = not on; t = t2
    b,g,r,a = color_bgra
    for a0,a1 in segs:
        cv2.line(img_bgra, a0, a1, (b,g,r,a), thickness, lineType=cv2.LINE_AA)

def render_lasso_overlay_png(image_shape: Tuple[int,int, int], vectors: Dict[str, List[List[Tuple[int,int]]]]) -> Tuple[str, str]:
    H, W = image_shape[:2]
    overlay = np.zeros((H, W, 4), np.uint8)  # BGRA

    fill_alpha = 46  # ~0.18 * 255
    line_alpha = 255
    thickness = 2

    for name, polys in vectors.items():
        if not isinstance(polys, list) or name == "person":
            continue
        color_hex = _PALETTE_HEX.get(name, "#b0b0b0")
        bgr = _hex_to_bgr(color_hex)

        for poly in polys:
            if len(poly) < 3: continue
            pts = np.array(poly, np.int32).reshape((-1,1,2))
            cv2.fillPoly(overlay, [pts], (bgr[0], bgr[1], bgr[2], fill_alpha))

        for poly in polys:
            if len(poly) < 2: continue
            _draw_dashed_polyline(overlay, poly, (255,255,255,line_alpha), dash_px=8, gap_px=8, thickness=thickness)

    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        return "", ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return b64, f"data:image/png;base64,{b64}"

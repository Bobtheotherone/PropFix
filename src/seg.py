from __future__ import annotations
import cv2, numpy as np

MP_OK = True
try:
    import mediapipe as mp
except Exception:
    MP_OK = False


def _mask_selfie(img_bgr):
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = seg.process(rgb)
    seg.close()
    if res.segmentation_mask is None:
        m = np.ones(img_bgr.shape[:2], np.float32)
    else:
        m = res.segmentation_mask.astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), 3.0)
    return np.clip(m, 0.0, 1.0)


def _mask_grabcut(img_bgr, rect):
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = rect
    x0 = max(0, min(w - 2, int(x0)))
    x1 = max(x0 + 1, min(w - 1, int(x1)))
    y0 = max(0, min(h - 2, int(y0)))
    y1 = max(y0 + 1, min(h - 1, int(y1)))
    rect = (x0, y0, x1 - x0, y1 - y0)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mask = np.zeros((h, w), np.uint8)
    cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    return np.clip(cv2.GaussianBlur(m, (0, 0), 1.5), 0.0, 1.0)


def torso_rect_from_pts(pts):
    sl, sr = pts["shoulder_l"], pts["shoulder_r"]
    hl, hr = pts["hip_l"], pts["hip_r"]
    shoulder_y = 0.5 * (sl[1] + sr[1])
    hip_y = 0.5 * (hl[1] + hr[1])
    span = abs(hip_y - shoulder_y)
    x_left = min(sl[0], hl[0]) - 0.8 * abs(sr[0] - sl[0])
    x_right = max(sr[0], hr[0]) + 0.8 * abs(sr[0] - sl[0])
    y_top = min(sl[1], sr[1]) - 0.8 * span
    y_bot = max(hl[1], hr[1]) + 0.8 * span
    return (x_left, y_top, x_right, y_bot)


def build_person_mask(img_bgr, pts, backend="auto"):
    rect = torso_rect_from_pts(pts)
    if backend in ("auto", "mp") and MP_OK:
        m = _mask_selfie(img_bgr)
        h, w = m.shape
        x0, y0, x1, y1 = int(max(0, rect[0])), int(max(0, rect[1])), int(min(w - 1, rect[2])), int(min(h - 1, rect[3]))
        roi = m[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
        if roi.size > 0 and float(np.mean(roi)) > 0.05:
            return m
    return _mask_grabcut(img_bgr, rect)


def _draw_capsule(mask, a, b, r):
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    cv2.circle(mask, (ax, ay), r, 0.0, -1, cv2.LINE_AA)
    cv2.circle(mask, (bx, by), r, 0.0, -1, cv2.LINE_AA)
    cv2.line(mask, (ax, ay), (bx, by), 0.0, thickness=r * 2, lineType=cv2.LINE_AA)


def suppress_arms_hands(mask01, pts, frac_radius=0.12):
    m = mask01.copy()
    sl, sr = pts["shoulder_l"], pts["shoulder_r"]
    sh_span = max(10.0, float(sr[0] - sl[0]))
    r = int(frac_radius * sh_span)
    if "elbow_l" in pts and "wrist_l" in pts:
        _draw_capsule(m, pts["elbow_l"], pts["wrist_l"], r)
    if "elbow_r" in pts and "wrist_r" in pts:
        _draw_capsule(m, pts["elbow_r"], pts["wrist_r"], r)
    for c in pts.get("wrists", []):
        cv2.circle(m, (int(c[0]), int(c[1])), max(4, r // 2 if r > 0 else 6), 0.0, -1, cv2.LINE_AA)
    return m


def torso_only_mask(
    person01,
    pts,
    tighten: float = 0.88,
    vtop_frac: float = 0.10,
    vbot_frac: float = 0.08,
    feather_px: int = 16,
):
    """
    Build a torso-only mask with strict vertical gating and soft top/bottom feathering.

    vtop_frac: start *below* shoulder row (fraction of shoulder–hip span)
    vbot_frac: end   *above* hip row      (fraction of shoulder–hip span)
    tighten:   lateral clamp vs shoulder half-width (0..1)
    feather_px: vertical cosine feather (pixels) to avoid hard seams
    """
    H, W = person01.shape
    sl, sr = pts["shoulder_l"], pts["shoulder_r"]
    hl, hr = pts["hip_l"], pts["hip_r"]
    shoulder_y = 0.5 * (sl[1] + sr[1])
    hip_y = 0.5 * (hl[1] + hr[1])
    span = abs(hip_y - shoulder_y)

    # STRICT vertical band: below shoulders and above hips
    top = int(round(shoulder_y + vtop_frac * span))
    bot = int(round(hip_y - vbot_frac * span))
    top = max(0, min(H - 2, top))
    bot = max(top + 1, min(H - 1, bot))

    # Lateral clamp around torso center
    midx = 0.5 * (sl[0] + sr[0] + hl[0] + hr[0]) * 0.5
    half = 0.5 * (sr[0] - sl[0])
    gate_half = tighten * max(half, 10.0)

    m = np.zeros_like(person01, np.float32)
    m[top : bot + 1, :] = person01[top : bot + 1, :]

    xs = np.arange(W)[None, :].astype(np.float32)
    lat_gate = ((xs >= (midx - gate_half)) & (xs <= (midx + gate_half))).astype(np.float32)
    m *= lat_gate

    # Vertical cosine feather to avoid seams at top/bottom
    if feather_px > 0:
        v = np.zeros((H, 1), np.float32)
        v[top : bot + 1, 0] = 1.0
        tband = min(feather_px, bot - top + 1)
        if tband > 0:
            tt = np.linspace(0, np.pi, tband, dtype=np.float32)
            v[top : top + tband, 0] = 0.5 * (1 - np.cos(tt))
            v[bot - tband + 1 : bot + 1, 0] = 0.5 * (1 - np.cos(tt[::-1]))
        m *= v

    # Clean small specks
    k = max(1, int(round(0.01 * W)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, 1)

    return np.clip(m, 0.0, 1.0)


def shoulder_dilate(mask01, pts, band_frac=0.06, px=None):
    H, W = mask01.shape
    sl, sr = pts["shoulder_l"], pts["shoulder_r"]
    hl, hr = pts["hip_l"], pts["hip_r"]
    shoulder_y = int(round(0.5 * (sl[1] + sr[1])))
    hip_y = int(round(0.5 * (hl[1] + hr[1])))
    span = abs(hip_y - shoulder_y)
    band = max(1, int(round(band_frac * span)))
    top = max(0, shoulder_y - band)
    bot = min(H - 1, shoulder_y + band)
    if px is None:
        px = max(2, int(round(0.02 * W)))
    patch = mask01[top : bot + 1, :]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    patch = cv2.dilate(patch, ker, 1)
    out = mask01.copy()
    out[top : bot + 1, :] = patch
    return out

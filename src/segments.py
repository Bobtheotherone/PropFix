from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import cv2

@dataclass
class Segment:
    name: str
    mask: np.ndarray  # float32 [0,1], HxW
    center: Tuple[float, float]
    ex: Tuple[float, float]  # local width axis (unit)
    ey: Tuple[float, float]  # local length axis (unit)
    color: Tuple[int, int, int]  # BGR


SEGMENT_ORDER = [
    "head", "neck", "torso",
    "upper_arm_l", "upper_arm_r",
    "forearm_l", "forearm_r",
    "hand_l", "hand_r",
    "thigh_l", "thigh_r",
    "shin_l", "shin_r",
    "foot_l", "foot_r",
]

_PALETTE = {
    "head": (40, 160, 255),
    "neck": (80, 80, 255),
    "torso": (0, 200, 80),
    "upper_arm_l": (220, 160, 60),
    "upper_arm_r": (60, 180, 220),
    "forearm_l": (220, 80, 80),
    "forearm_r": (80, 120, 220),
    "hand_l": (160, 60, 200),
    "hand_r": (60, 60, 180),
    "thigh_l": (160, 220, 60),
    "thigh_r": (60, 220, 160),
    "shin_l": (180, 80, 160),
    "shin_r": (120, 60, 160),
    "foot_l": (80, 180, 120),
    "foot_r": (60, 160, 120),
}

# ----------------- helpers -----------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1])) + 1e-9
    return v / n


def _perp(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=np.float32)


def _soften(mask: np.ndarray, ks: int) -> np.ndarray:
    ks = int(max(3, ks | 1))
    m = cv2.GaussianBlur(mask, (ks, ks), 0)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def _fill_poly(shape, pts: np.ndarray, blur_px=11) -> np.ndarray:
    H, W = shape[:2]
    m = np.zeros((H, W), np.float32)
    cv2.fillConvexPoly(m, pts.astype(np.int32), 1.0)
    return _soften(m, blur_px)


def _fill_circle(shape, c, r, blur_px=11) -> np.ndarray:
    H, W = shape[:2]
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (int(c[0]), int(c[1])), int(max(1.0, r)), 1.0, -1, cv2.LINE_AA)
    return _soften(m, blur_px)


def _bone(a, b) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    a = np.array(a, np.float32)
    b = np.array(b, np.float32)
    v = b - a
    L = float(np.hypot(v[0], v[1])) + 1e-6
    ey = _unit(v)  # along-bone (length)
    ex = _perp(ey)  # across-bone (width)
    return a, L, ex, ey


def _quad_along(a, b, half_w: float) -> np.ndarray:
    a = np.array(a, np.float32)
    b = np.array(b, np.float32)
    _, _, ex, ey = _bone(a, b)
    p0 = a + ex * half_w
    p1 = b + ex * half_w
    p2 = b - ex * half_w
    p3 = a - ex * half_w
    return np.stack([p0, p1, p2, p3], axis=0)


def _rect_centered(c, ex, ey, w, h) -> np.ndarray:
    """Rotated rectangle polygon centered at c with axes ex(width)/ey(length)."""
    ex = _unit(np.asarray(ex, np.float32))
    ey = _unit(np.asarray(ey, np.float32))
    c = np.asarray(c, np.float32)
    dx = ex * (0.5 * w)
    dy = ey * (0.5 * h)
    p0 = c - dx - dy
    p1 = c + dx - dy
    p2 = c + dx + dy
    p3 = c - dx + dy
    return np.stack([p0, p1, p2, p3], axis=0)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask > 1e-3)
    if xs.size == 0:
        H, W = mask.shape
        return W * 0.5, H * 0.5
    return float(xs.mean()), float(ys.mean())


# ----------------- segment construction -----------------
def build_segments_from_landmarks(shape, pts: Dict[str, tuple]) -> Dict[str, Segment]:
    H, W = shape[:2]
    segs: Dict[str, Segment] = {}

    need = ("shoulder_l", "shoulder_r", "hip_l", "hip_r")
    if not all(k in pts for k in need):
        return segs

    sl = np.array(pts["shoulder_l"], np.float32)
    sr = np.array(pts["shoulder_r"], np.float32)
    hl = np.array(pts["hip_l"], np.float32)
    hr = np.array(pts["hip_r"], np.float32)

    shoulder_span = float(np.hypot(*(sr - sl))) + 1e-6
    pelvis_span = float(np.hypot(*(hr - hl))) + 1e-6
    span = max(shoulder_span, pelvis_span)
    shoulder_y = float(0.5 * (sl[1] + sr[1]))
    mid_s = 0.5 * (sl + sr)

    # ---- Torso ----
    torso_poly = np.stack([sl, sr, hr, hl], axis=0)
    torso_m = _fill_poly((H, W, 3), torso_poly, blur_px=17)
    cx, cy = _centroid(torso_m)
    segs["torso"] = Segment("torso", torso_m, (cx, cy), (1.0, 0.0), (0.0, 1.0), _PALETTE["torso"])

    # ===================== HEAD (cheek-centered; aligned to forehead→chin) =====================
    head_w = None
    head_h = None
    head_center = None
    ex_face = None
    ey_face = None

    if all(k in pts for k in ("forehead", "chin", "face_left", "face_right")):
        F = np.array(pts["forehead"], np.float32)
        C = np.array(pts["chin"], np.float32)
        L = np.array(pts["face_left"], np.float32)
        R = np.array(pts["face_right"], np.float32)

        v = C - F
        ey_face = _unit(v)  # downwards axis along face height
        ex_face = _perp(ey_face)  # left–right across face
        # ensure ex points from left cheek to right cheek
        if np.dot(R - L, ex_face) < 0:
            ex_face = -ex_face

        head_h = float(np.linalg.norm(v))
        head_w = float(abs(np.dot(R - L, ex_face)))

        # base center along forehead→chin, then recenter along ex to cheek midpoint
        center = F + 0.46 * v  # 0.46 positions center slightly below mid-face
        cheek_mid = 0.5 * (L + R)
        # replace the ex-component with that of the cheek midpoint (horizontal centering)
        ex_delta = np.dot(cheek_mid - center, ex_face)
        head_center = center + ex_face * ex_delta

        # ellipse radii (tightened to avoid hair/ears)
        rx = 0.95 * (0.5 * head_w)  # semi-axis along ex
        ry = 0.88 * (0.5 * head_h)  # semi-axis along ey

        # draw ellipse
        ang = float(np.degrees(np.arctan2(ex_face[1], ex_face[0])))
        mask = np.zeros((H, W), np.uint8)
        cv2.ellipse(
            mask,
            (int(head_center[0]), int(head_center[1])),
            (int(max(1.0, rx)), int(max(1.0, ry))),
            ang, 0, 360, 255, -1, cv2.LINE_AA
        )
        head_m = _soften(mask.astype(np.float32) / 255.0, 15)
        hcx, hcy = _centroid(head_m)
        segs["head"] = Segment("head", head_m, (hcx, hcy), (1.0, 0.0), (0.0, 1.0), _PALETTE["head"])
    else:
        # Fallback: circle above shoulders
        head_center = np.array([float(mid_s[0]), float(mid_s[1] - 0.55 * span)], np.float32)
        r = 0.30 * span
        ex_face = np.array([1.0, 0.0], np.float32)
        ey_face = np.array([0.0, 1.0], np.float32)
        head_w = head_h = 2.0 * r
        head_m = _fill_circle((H, W, 3), head_center, r, blur_px=15)
        hcx, hcy = _centroid(head_m)
        segs["head"] = Segment("head", head_m, (hcx, hcy), (1.0, 0.0), (0.0, 1.0), _PALETTE["head"])

    # ===================== NECK (Fixed alignment) =====================
    chin_pt = np.array(pts.get("chin", head_center + ey_face * (0.5 * head_h)), np.float32)
    s_mid = 0.5 * (sl + sr)  # Shoulder midpoint

    # Calculate vector from chin to shoulders and neck dimensions
    v = s_mid - chin_pt
    neck_h_val = max(0.01, np.dot(v, ey_face))  # Height along face axis
    v_ex_val = np.dot(v, ex_face)  # Lateral offset along face axis

    # Position neck center to connect head and torso
    neck_center = chin_pt + (0.5 * neck_h_val) * ey_face + v_ex_val * ex_face
    neck_w_val = 0.45 * shoulder_span  # Width proportional to shoulder span

    # Create neck polygon and mask
    neck_poly = _rect_centered(neck_center, ex_face, ey_face, neck_w_val, neck_h_val)
    neck_m = _fill_poly((H, W, 3), neck_poly, blur_px=13)
    ncx, ncy = _centroid(neck_m)
    segs["neck"] = Segment("neck", neck_m, (ncx, ncy), tuple(ex_face.tolist()), tuple(ey_face.tolist()), _PALETTE["neck"])

    # ---- Arms ----
    def _maybe(name_a, name_b, half_scale, color_key, seg_name):
        if name_a in pts and name_b in pts:
            a, b = pts[name_a], pts[name_b]
            _, L, exL, eyL = _bone(a, b)
            m = _fill_poly((H, W, 3), _quad_along(a, b, float(half_scale * L)), 11)
            cx_, cy_ = _centroid(m)
            segs[seg_name] = Segment(seg_name, m, (cx_, cy_), tuple(exL), tuple(eyL), _PALETTE[color_key])

    _maybe("shoulder_l", "elbow_l", 0.14, "upper_arm_l", "upper_arm_l")
    _maybe("shoulder_r", "elbow_r", 0.14, "upper_arm_r", "upper_arm_r")
    _maybe("elbow_l", "wrist_l", 0.12, "forearm_l", "forearm_l")
    _maybe("elbow_r", "wrist_r", 0.12, "forearm_r", "forearm_r")

    if "wrist_l" in pts:
        hm = _fill_circle((H, W, 3), pts["wrist_l"], 0.10 * span, 9)
        segs["hand_l"] = Segment("hand_l", hm, _centroid(hm), (1.0, 0.0), (0.0, 1.0), _PALETTE["hand_l"])
    if "wrist_r" in pts:
        hm = _fill_circle((H, W, 3), pts["wrist_r"], 0.10 * span, 9)
        segs["hand_r"] = Segment("hand_r", hm, _centroid(hm), (1.0, 0.0), (0.0, 1.0), _PALETTE["hand_r"])

    # ---- Legs ----
    _maybe("hip_l", "knee_l", 0.18, "thigh_l", "thigh_l")
    _maybe("hip_r", "knee_r", 0.18, "thigh_r", "thigh_r")
    _maybe("knee_l", "ankle_l", 0.14, "shin_l", "shin_l")
    _maybe("knee_r", "ankle_r", 0.14, "shin_r", "shin_r")

    if "ankle_l" in pts:
        fm = _fill_circle((H, W, 3), pts["ankle_l"], 0.12 * span, 9)
        segs["foot_l"] = Segment("foot_l", fm, _centroid(fm), (1.0, 0.0), (0.0, 1.0), _PALETTE["foot_l"])
    if "ankle_r" in pts:
        fm = _fill_circle((H, W, 3), pts["ankle_r"], 0.12 * span, 9)
        segs["foot_r"] = Segment("foot_r", fm, _centroid(fm), (1.0, 0.0), (0.0, 1.0), _PALETTE["foot_r"])

    return segs


# ----------------- visualization -----------------
def draw_annotation_overlay(
        img_bgr: np.ndarray,
        segments: Dict[str, Segment],
        alpha: float = 0.35,
        draw_edges: bool = True,
        draw_keypoints: Dict[str, tuple] | None = None,
) -> np.ndarray:
    """Color overlay per segment with optional edges and keypoints."""
    H, W = img_bgr.shape[:2]
    color_img = np.zeros_like(img_bgr, np.float32)

    for name in SEGMENT_ORDER:
        seg = segments.get(name)
        if seg is None:
            continue
        c = np.array(seg.color, np.float32)[None, None, :]
        m = seg.mask[..., None].astype(np.float32)
        color_img += m * c

    color_img = np.clip(color_img, 0, 255).astype(np.uint8)
    blended = cv2.addWeighted(img_bgr, 1.0, color_img, alpha, 0)

    if draw_edges:
        edge = np.zeros((H, W), np.uint8)
        for seg in segments.values():
            e = cv2.Canny((seg.mask * 255).astype(np.uint8), 40, 120)
            edge = np.maximum(edge, e)
        blended[edge > 0] = (0.6 * blended[edge > 0] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)

    if draw_keypoints:
        for k, (x, y) in draw_keypoints.items():
            cv2.circle(blended, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(blended, k, (int(x) + 4, int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return blended

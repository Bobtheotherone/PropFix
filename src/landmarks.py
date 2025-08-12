from __future__ import annotations
import cv2
import numpy as np

MP_OK = True
try:
    import mediapipe as mp
except Exception:
    MP_OK = False


def _to_px(pt, w: int, h: int) -> tuple[int, int]:
    x = int(round(pt.x * w))
    y = int(round(pt.y * h))
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


def detect_landmarks(img_bgr: np.ndarray) -> dict:
    """
    Returns pixel coordinates for body joints and face anchors.
    Keys (subset may be missing):
      face_left/right (cheeks), head_left/right (aliases),
      forehead, chin, nose, face_oval (list of (x,y)),
      shoulder_l/r, elbow_l/r, wrist_l/r, hip_l/r, knee_l/r, ankle_l/r
    """
    if not MP_OK:
        return {}

    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out: dict[str, object] = {}

    # ---------- FaceMesh (for head ellipse + neck axis) ----------
    try:
        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        )
        fr = fm.process(rgb)
        fm.close()
        if fr.multi_face_landmarks:
            lmk = fr.multi_face_landmarks[0].landmark

            # Cheeks (outer)
            li, ri = 234, 454
            lx, ly = _to_px(lmk[li], w, h)
            rx, ry = _to_px(lmk[ri], w, h)
            if lx > rx:
                lx, rx, ly, ry = rx, lx, ry, ly
            out["face_left"] = (lx, ly)
            out["face_right"] = (rx, ry)
            out["head_left"] = (lx, ly)
            out["head_right"] = (rx, ry)

            # Chin & forehead anchors
            chin_id, fore_id = 152, 10
            out["chin"] = _to_px(lmk[chin_id], w, h)
            out["forehead"] = _to_px(lmk[fore_id], w, h)

            # Face-oval ring (robust ellipse fitting)
            # Common set used in MediaPipe examples:
            face_oval_ids = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            out["face_oval"] = [_to_px(lmk[i], w, h) for i in face_oval_ids if 0 <= i < len(lmk)]
    except Exception:
        pass

    # ---------- Pose (full body) ----------
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pr = pose.process(rgb)
    pose.close()
    if pr.pose_landmarks:
        lm = pr.pose_landmarks.landmark
        vis_thr = 0.35

        def P(i: int):
            if getattr(lm[i], "visibility", 1.0) < vis_thr:
                return None
            return _to_px(lm[i], w, h)

        idx = {
            "nose": 0,
            "shoulder_l": 11, "shoulder_r": 12,
            "elbow_l": 13, "elbow_r": 14,
            "wrist_l": 15, "wrist_r": 16,
            "hip_l": 23, "hip_r": 24,
            "knee_l": 25, "knee_r": 26,
            "ankle_l": 27, "ankle_r": 28,
        }
        for k, i in idx.items():
            p = P(i)
            if p is not None:
                out[k] = p

    return out

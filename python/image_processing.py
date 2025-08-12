import sys, os, json
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Simple helpers --------------------------------------------------------------
def abspath(p: str) -> str:
    return str(Path(p).expanduser().resolve())

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_image(img: Image.Image, out_path: Path) -> str:
    ensure_dir(out_path.parent)
    img.save(out_path)
    return str(out_path.resolve())

def load_image(path_str: str) -> Image.Image:
    return Image.open(path_str).convert("RGB")

# "Segmentation" stub (returns plausible parts) -------------------------------
DEFAULT_SEGMENTS = [
    "torso", "head", "l_arm", "r_arm", "l_forearm", "r_forearm",
    "l_thigh", "r_thigh", "l_calf", "r_calf", "l_hand", "r_hand",
    "l_foot", "r_foot"
]

def annotate(img: Image.Image, segments: List[str]) -> Image.Image:
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    # draw light grid rectangles where labels go
    n = max(4, int(np.sqrt(len(segments))))
    cell_w, cell_h = w // n, h // n
    idx = 0
    for r in range(n):
        for c in range(n):
            if idx >= len(segments):
                break
            x0, y0 = c * cell_w + 4, r * cell_h + 4
            x1, y1 = (c + 1) * cell_w - 4, (r + 1) * cell_h - 4
            draw.rectangle([x0, y0, x1, y1], outline=(79,142,247,180), width=2)
            label = segments[idx].replace("_", " ")
            draw.text((x0+6, y0+6), label, fill=(230,232,238,220))
            idx += 1
    return img

# Simple "warp" (global transform) -------------------------------------------
def warp_image(img: Image.Image, controls: Dict[str, Any]) -> Image.Image:
    # We approximate part controls with one global scale + translate + rotate
    geo = (controls or {}).get("geometry", {})
    # Derive average scale/offset
    sx_vals, sy_vals, tx_vals, ty_vals, rot_vals = [], [], [], [], []
    for part, g in geo.items():
        sx_vals.append(float(g.get("sx", 1.0)))
        sy_vals.append(float(g.get("sy", 1.0)))
        tx_vals.append(float(g.get("tx", 0.0)))
        ty_vals.append(float(g.get("ty", 0.0)))
        rot_vals.append(float(g.get("rot_deg", 0.0)))
    def avg(vs, default):
        return (sum(vs) / len(vs)) if vs else default
    sx = max(0.6, min(1.4, avg(sx_vals, 1.0)))
    sy = max(0.6, min(1.4, avg(sy_vals, 1.0)))
    tx = avg(tx_vals, 0.0)
    ty = avg(ty_vals, 0.0)
    rot = avg(rot_vals, 0.0)

    # Apply rotate then scale then translate
    w, h = img.size
    img2 = img.rotate(-rot, resample=Image.BICUBIC, expand=False, center=(w/2, h/2))
    new_w, new_h = int(w * sx), int(h * sy)
    img2 = img2.resize((new_w, new_h), resample=Image.BICUBIC)

    # Paste onto a canvas of original size, centered with offset
    canvas = Image.new("RGB", (w, h), (0,0,0))
    off_x = int((w - new_w)//2 + tx)
    off_y = int((h - new_h)//2 + ty)
    canvas.paste(img2, (off_x, off_y))
    return canvas

# Main entry ------------------------------------------------------------------
def main(argv: List[str]):
    # Accept JSON payload via argv[1]
    payload: Dict[str, Any] = {}
    if len(argv) >= 2:
        try:
            payload = json.loads(argv[1])
        except Exception:
            payload = {}

    mode = str(payload.get("mode", "classify")).lower()
    input_path = payload.get("input") or ""
    output_path = payload.get("output")  # may be None; we'll decide
    if not input_path:
        print(json.dumps({"ok": False, "error": "Missing input path"}))
        return

    in_abs = abspath(input_path)
    try:
        img = load_image(in_abs)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Failed to open image: {e}"}))
        return

    root = Path(__file__).resolve().parent.parent
    outputs = root / "outputs"
    ensure_dir(outputs)

    if mode == "classify":
        segs = DEFAULT_SEGMENTS.copy()
        annotated = annotate(img.copy(), segs)
        out = Path(output_path) if output_path else outputs / (Path(in_abs).stem + "_annotated.png")
        out_abs = save_image(annotated, out)
        print(json.dumps({"ok": True, "mode": mode, "segments": segs, "used_segments": segs, "output": out_abs}))
        return

    elif mode == "warp":
        controls = payload.get("controls", {})
        warped = warp_image(img.copy(), controls)
        out = Path(output_path) if output_path else outputs / (Path(in_abs).stem + "_warped.png")
        out_abs = save_image(warped, out)
        print(json.dumps({"ok": True, "mode": mode, "output": out_abs}))
        return

    elif mode in ("enhance", "denoise", "style"):
        # Placeholder: just save copy with a suffix
        suffix = {"enhance": "_enhanced", "denoise": "_denoised", "style": "_styled"}[mode]
        out = Path(output_path) if output_path else outputs / (Path(in_abs).stem + f"{suffix}.png")
        out_abs = save_image(img.copy(), out)
        print(json.dumps({"ok": True, "mode": mode, "output": out_abs}))
        return

    else:
        print(json.dumps({"ok": False, "error": f"Unknown mode: {mode}"}))
        return

if __name__ == "__main__":
    main(sys.argv)

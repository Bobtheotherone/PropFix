# server/propfix_server.py
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.resolve()        # repo root
PYEXE = sys.executable                     # use the active conda env's python
DEFAULT_CFG = ROOT / "configs" / "default.yaml"
SERVER_OUT = ROOT / "server" / "outputs"
SERVER_OUT.mkdir(parents=True, exist_ok=True)

# Ensure 'src' is importable for subprocesses regardless of CWD
os.environ.setdefault("PYTHONPATH", str(ROOT))

app = Flask(__name__)

def _abs(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q).resolve()

def _choose_output(req_output: Optional[str], default_name: str) -> Path:
    if req_output:
        q = Path(req_output)
        if q.is_absolute():
            q.parent.mkdir(parents=True, exist_ok=True)
            return q
        out = (SERVER_OUT / q).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out = (SERVER_OUT / default_name).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def _segments_to_names(raw) -> list[str]:
    names: list[str] = []
    if isinstance(raw, list):
        for i, s in enumerate(raw):
            if isinstance(s, str):
                names.append(s)
            elif isinstance(s, dict):
                nm = s.get("name") or s.get("id") or s.get("label")
                names.append(str(nm) if nm else f"seg{i}")
            else:
                names.append(f"seg{i}")
    return names

def _adapt_controls(ctrl: Dict[str, Any]) -> Dict[str, Any]:
    """Return a controls dict augmented with common synonyms so that
    older/newer warp implementations can read the geometry.
    """
    if not isinstance(ctrl, dict):
        return {"geometry": {}, "photo": {}}
    geom = dict(ctrl.get("geometry") or {})
    photo = dict(ctrl.get("photo") or {})

    adapted = {}
    for seg, g in geom.items():
        if not isinstance(g, dict):
            continue
        sx = float(g.get("sx", g.get("scale_x", g.get("scaleX", 1.0)) or 1.0))
        sy = float(g.get("sy", g.get("scale_y", g.get("scaleY", 1.0)) or 1.0))
        rot = float(g.get("rot_deg", g.get("rotation_deg", g.get("angle_deg", g.get("theta_deg", 0.0))) or 0.0))
        tx = float(g.get("tx", g.get("dx", 0.0)) or 0.0)
        ty = float(g.get("ty", g.get("dy", 0.0)) or 0.0)

        gg = {
            "sx": sx, "sy": sy, "rot_deg": rot, "tx": tx, "ty": ty,
            # common synonyms many warp modules use
            "scale_x": sx, "scale_y": sy, "scaleX": sx, "scaleY": sy,
            "rotation_deg": rot, "angle_deg": rot, "theta_deg": rot,
            "dx": tx, "dy": ty,
            "scale": {"x": sx, "y": sy},
            "translate": {"x": tx, "y": ty},
            "rotate": {"deg": rot},
        }
        adapted[seg] = gg

    return {"geometry": adapted, "photo": photo, "echo_from_server": True}

def _run_cli(mode: str, cfg: Path, inp: Path, out: Path,
             manual_json: Optional[Path], controls_json: Optional[Path],
             timeout_s: float = 120.0) -> Dict[str, Any]:
    import subprocess

    cmd = [
        PYEXE, "-m", "src.cli",
        "--mode", mode,
        "--config", str(cfg),
        "--input", str(inp),
        "--output", str(out),
    ]
    if manual_json:
        cmd += ["--manual_json", str(manual_json)]
    if controls_json:
        cmd += ["--controls_json", str(controls_json)]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),                    # critical: import src.*
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    dt = time.time() - t0

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    stderr_tail = "\n".join(stderr.splitlines()[-40:]) if stderr else ""

    payload: Dict[str, Any] = {}
    try:
        payload = json.loads(stdout) if stdout else {}
    except Exception:
        payload = {"stdout_raw": stdout}

    # Normalize and enrich payload
    if "segments" not in payload and isinstance(payload.get("used_segments"), list):
        payload["segments"] = payload["used_segments"]
    payload["segments"] = _segments_to_names(payload.get("segments"))
    if not payload.get("output"):
        payload["output"] = str(out)
    payload.setdefault("op", mode)
    payload.setdefault("elapsed_sec", round(dt, 3))
    if stderr_tail:
        payload["stderr_tail"] = stderr_tail
    payload["rc"] = proc.returncode
    return payload

@app.get("/health")
def health():
    return jsonify({"ok": True, "root": str(ROOT)})

@app.post("/process")
def process():
    req = request.get_json(force=True, silent=False) or {}

    mode = req.get("mode", "classify")
    if mode not in ("classify", "warp"):
        return jsonify({"error": f"invalid mode: {mode}"}), 400

    cfg = _abs(req.get("config")) or DEFAULT_CFG
    inp = _abs(req.get("input"))
    if not inp or not inp.exists():
        return jsonify({"error": f"input not found: {req.get('input')!r}"}), 400

    out = _choose_output(req.get("output"), f"preview_{mode}.png")
    manual_json = _abs(req.get("manual_json"))

    controls_json = _abs(req.get("controls_json"))
    tmp_controls_path: Optional[Path] = None
    if mode == "warp":
        # Always adapt inline controls (or empty) to maximize compatibility
        incoming = req.get("controls") if isinstance(req.get("controls"), dict) else {"geometry": {}, "photo": {}}
        adapted = _adapt_controls(incoming)
        tmp_controls_path = out.with_suffix(".controls.json")
        tmp_controls_path.write_text(json.dumps(adapted, ensure_ascii=False, indent=2), encoding="utf-8")
        controls_json = tmp_controls_path

    payload = _run_cli(mode, cfg, inp, out, manual_json, controls_json)

    if tmp_controls_path and tmp_controls_path.exists() and payload.get("rc") == 0:
        try:
            tmp_controls_path.unlink(missing_ok=True)
        except Exception:
            pass

    status = 200 if payload.get("rc", 1) == 0 else 500
    return jsonify(payload), status

if __name__ == "__main__":
    host = os.environ.get("PROPFIX_HOST", "127.0.0.1")
    port = int(os.environ.get("PROPFIX_PORT", "5001"))
    app.run(host=host, port=port, debug=False)

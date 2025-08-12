"""Simple Python back‑end for PropFix.

This script reads a JSON command from standard input, applies an image
processing operation and writes the result path or base64 string to
standard output.  It is intentionally minimal to demonstrate
communication between Electron/Node and Python using the `python‑shell`
package【362342226299824†L68-L82】.

Supported commands:

* ``enhance`` – automatically adjust contrast using histogram equalisation.
* ``denoise`` – apply a median filter to remove noise.
* ``style`` – placeholder for style transfer.  Currently returns the
  original image.

The script can be extended to support additional operations by adding
handlers to the ``COMMANDS`` dictionary.
"""
import json
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image


def enhance_image(img: np.ndarray) -> np.ndarray:
    """Enhance image contrast using histogram equalisation."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    enhanced = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return enhanced


def denoise_image(img: np.ndarray) -> np.ndarray:
    """Reduce noise using a median filter."""
    return cv2.medianBlur(img, 3)


def style_transfer_image(img: np.ndarray) -> np.ndarray:
    """Placeholder for style transfer – returns the original image."""
    return img


COMMANDS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "enhance": enhance_image,
    "denoise": denoise_image,
    "style": style_transfer_image,
}


def process_request(request: dict) -> str:
    """Process a JSON command and return a path to the processed image.

    Parameters
    ----------
    request : dict
        Dictionary with keys ``command``, ``image_path`` and optional
        ``params``.

    Returns
    -------
    str
        Path to the processed image.  The image is saved in the same
        directory as the input with suffix ``_propfix``.
    """
    cmd = request.get("command")
    img_path = Path(request.get("image_path"))
    if cmd not in COMMANDS:
        raise ValueError(f"Unsupported command: {cmd}")
    # Load image using OpenCV (BGR format)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    processed = COMMANDS[cmd](img)
    # Save result
    out_path = img_path.with_name(img_path.stem + "_propfix" + img_path.suffix)
    cv2.imwrite(str(out_path), processed)
    return str(out_path)


def main() -> None:
    # Read entire message from stdin (python‑shell sends a JSON string)
    data = sys.stdin.read().strip()
    if not data:
        return
    try:
        request = json.loads(data)
        output_path = process_request(request)
        # Print the path back to Node/Electron
        print(output_path)
    except Exception as exc:
        # In case of error, print error message so Node can handle it
        print(f"error: {exc}")
    finally:
        sys.stdout.flush()


if __name__ == "__main__":
    main()
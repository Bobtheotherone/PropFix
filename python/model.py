"""Model definitions for PropFix.

This module contains wrappers around deep‑learning models used in PropFix.
The classes here provide a uniform API for loading weights, preprocessing
input images and generating predictions.  They are intentionally kept
simple and can be extended to include more sophisticated models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import cv2

try:
    import torch
    import torchvision.transforms as T
    from torch import nn
except ImportError:
    # Torch is optional; fallback if not available.
    torch = None
    nn = None


@dataclass
class BaseModel:
    """Base class for models.  All subclasses must implement `predict`."""

    name: str

    def load(self, weight_path: str) -> None:
        """Load model weights from a file.  Subclasses may override this."""
        raise NotImplementedError

    def preprocess(self, image: np.ndarray) -> Any:
        """Preprocess an image before feeding it to the model."""
        # Default implementation normalises values to [0, 1]
        return image.astype(np.float32) / 255.0

    def postprocess(self, output: Any) -> np.ndarray:
        """Postprocess model output to an image array."""
        # Default implementation returns the output unchanged
        return output

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a single image and return the output image."""
        raise NotImplementedError


class DummySuperResolution(BaseModel):
    """A dummy super‑resolution model that simply resizes images.

    This class demonstrates the interface expected by the back‑end.
    Replace it with an actual neural network (e.g., ESRGAN) by
    implementing `load` and `predict` accordingly.
    """

    def __init__(self) -> None:
        super().__init__(name="dummy_super_resolution")

    def load(self, weight_path: str) -> None:
        # This dummy model does not use weights
        return

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Simple 2× nearest neighbour upsampling
        h, w = image.shape[:2]
        return cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)


class DummyDenoise(BaseModel):
    """A dummy denoising model that applies a Gaussian blur."""

    def __init__(self) -> None:
        super().__init__(name="dummy_denoise")

    def predict(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)

"""ArUco marker: make the picture and find it in camera images (OpenCV only, no Isaac).

The landing pad carries one marker of the dictionary ``DICT_4X4_50``: a 6 x 6 grid of cells, the outer ring black
(the border), the inner 4 x 4 cells a code that gives the id. ``detect`` returns, per image, where the marker is:

    center   pixel (u, v) of the middle of the marker, from the four corners
    side     mean length of the four edges [pixel], it shrinks with the distance to the marker
    found    False if no marker of this id is seen (too small, blurred, cut by the image border)

Formulas and the way the policy uses them: guide/06_aruco_landing.md.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
MARKER_TEXTURE = Path(__file__).resolve().parents[2] / "assets" / "data" / "aruco" / "aruco_4x4_id0.png"


def marker_image(marker_id: int = MARKER_ID, size_px: int = 600, border_px: int = 100) -> np.ndarray:
    """Grayscale picture (size_px x size_px, uint8) of the marker on a white margin ``border_px`` wide.

    The white margin is needed: the detector finds the black border of the marker against a light background.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    inner = size_px - 2 * border_px
    picture = np.full((size_px, size_px), 255, dtype=np.uint8)
    picture[border_px:border_px + inner, border_px:border_px + inner] = cv2.aruco.generateImageMarker(
        dictionary, marker_id, inner)
    return picture


def save_marker_texture(path: Path = MARKER_TEXTURE, marker_id: int = MARKER_ID) -> Path:
    """Write the texture that is put on the landing pad."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), marker_image(marker_id))
    return path


class ArucoDetector:
    """Detects the marker ``marker_id`` in a batch of RGB images."""

    def __init__(self, marker_id: int = MARKER_ID):
        parameters = cv2.aruco.DetectorParameters()
        parameters.minMarkerPerimeterRate = 0.02        # accept a marker down to about 2 % of the image width
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.marker_id = marker_id
        self._detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), parameters)

    def detect(self, images: np.ndarray) -> dict:
        """Images (N, H, W, 3 or 4) uint8 -> ``found`` (N,) bool, ``center`` (N, 2) [pixel], ``side`` (N,) [pixel]."""
        count = images.shape[0]
        found = np.zeros(count, dtype=bool)
        center = np.zeros((count, 2), dtype=np.float32)
        side = np.zeros(count, dtype=np.float32)
        for i in range(count):
            gray = cv2.cvtColor(np.ascontiguousarray(images[i, :, :, :3]), cv2.COLOR_RGB2GRAY)
            corners, ids, _ = self._detector.detectMarkers(gray)
            if ids is None:
                continue
            for corner, marker in zip(corners, ids.ravel()):
                if marker == self.marker_id:
                    quad = corner.reshape(4, 2)
                    found[i] = True
                    center[i] = quad.mean(axis=0)
                    side[i] = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1).mean()
                    break
        return {"found": found, "center": center, "side": side}


if __name__ == "__main__":
    print(f"Marker texture: {save_marker_texture()}")

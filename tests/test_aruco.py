import numpy as np

from Drone_RL.uav.mdp.aruco import ArucoDetector, marker_image


def _scene(marker_px: int, top_left: tuple[int, int], size: int = 160) -> np.ndarray:
    """Gray floor with the marker (white margin included) pasted at ``top_left``."""
    image = np.full((size, size), 120, dtype=np.uint8)
    picture = marker_image(size_px=600)
    import cv2
    picture = cv2.resize(picture, (marker_px, marker_px), interpolation=cv2.INTER_AREA)
    y, x = top_left
    image[y:y + marker_px, x:x + marker_px] = picture
    return np.repeat(image[None, :, :, None], 3, axis=3)


def test_marker_is_found_at_the_right_place_and_size():
    images = _scene(marker_px=60, top_left=(30, 50))
    result = ArucoDetector().detect(images)
    assert result["found"][0]
    # the marker without its white margin is 60 * 400 / 600 = 40 px wide, centred in the 60 px square
    assert np.allclose(result["center"][0], [50 + 30, 30 + 30], atol=1.5)
    assert abs(result["side"][0] - 40.0) < 2.0


def test_empty_floor_finds_nothing():
    images = np.full((2, 160, 160, 3), 120, dtype=np.uint8)
    assert not ArucoDetector().detect(images)["found"].any()

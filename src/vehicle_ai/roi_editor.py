"""Streamlit wrapper for a dependency-free browser canvas."""
import base64
from pathlib import Path

import cv2
import streamlit.components.v1 as components

_component = components.declare_component("vehicle_roi", path=str(Path(__file__).parent / "roi_component"))


def roi_editor(image, points, *, context, key):
    height, width = image.shape[:2]
    if width > 1280:
        image = cv2.resize(image, (1280, round(height * 1280 / width)))
    ok, data = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Không mã hóa được ảnh ROI.")
    encoded = base64.b64encode(data.tobytes()).decode("ascii")
    return _component(image="data:image/jpeg;base64," + encoded, points=points,
                      context=context, key=key, default=None)

"""Heuristic image ranking, not calibrated recognition probabilities."""
import math

import cv2
import numpy as np


def plate_quality(image, confidence, box, car_shape, *, target_width=100, target_height=32,
                  target_sharpness=150, edge_margin=2, expected_ratios=(1.4, 4.5)):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    x1, y1, x2, y2 = box
    car_height, car_width = car_shape[:2]
    margin = min(x1, y1, car_width - x2, car_height - y2)
    size_score = min(width / target_width, 1.0) * min(height / target_height, 1.0)
    sharp_score = min(sharpness / target_sharpness, 1.0)
    edge_score = max(0.0, min(margin / edge_margin, 1.0))
    ratio = width / max(height, 1)
    aspect_score = max(math.sqrt(min(ratio / expected, expected / ratio)) for expected in expected_ratios)
    score = float(np.clip(confidence, 0, 1)) * size_score * sharp_score * edge_score * aspect_score
    return {
        "quality_score": round(score, 6), "visual_quality_score": round(score, 6),
        "quality_factors": {
            "plate_width": width, "plate_height": height, "area_pixels": width * height,
            "sharpness": round(sharpness, 3), "edge_margin_pixels": round(margin, 3),
            "aspect_ratio": round(ratio, 3), "size_score": round(size_score, 4),
            "sharpness_score": round(sharp_score, 4), "edge_score": round(edge_score, 4),
            "aspect_score": round(aspect_score, 4),
        },
    }

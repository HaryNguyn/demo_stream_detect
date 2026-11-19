"""Normalized polygon ROI shared by review UI and inference."""
import math
import numpy as np
import cv2


def validate_roi(points):
    if points is None:
        return []
    if not isinstance(points, list):
        raise ValueError("ROI phải là danh sách tọa độ [x,y].")
    if not points:
        return []
    if len(points) < 3 or len(points) > 100:
        raise ValueError("ROI cần từ 3 đến 100 điểm.")
    clean = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Mỗi điểm ROI cần [x,y].")
        x, y = map(float, point)
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Tọa độ ROI phải trong [0,1].")
        clean.append([x, y])
    contour = np.asarray(clean, dtype=np.float32)
    if abs(cv2.contourArea(contour)) < 1e-6:
        raise ValueError("ROI phải có diện tích lớn hơn 0.")
    # Reject self-crossing polygons, including collinear non-adjacent edges.
    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and
                min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)

    def intersects(a, b, c, d):
        o1, o2, o3, o4 = orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        return any(abs(o) <= 1e-9 and on_segment(u, v, p) for o, u, v, p in (
            (o1, a, b, c), (o2, a, b, d), (o3, c, d, a), (o4, c, d, b)))

    for i in range(len(clean)):
        if clean[i] == clean[(i + 1) % len(clean)]:
            raise ValueError("ROI có hai đỉnh liên tiếp trùng nhau.")
        for j in range(i + 1, len(clean)):
            if j == i + 1 or (i == 0 and j == len(clean) - 1):
                continue
            if intersects(clean[i], clean[(i + 1) % len(clean)], clean[j], clean[(j + 1) % len(clean)]):
                raise ValueError("Các cạnh ROI không được tự cắt nhau.")
    return clean


def center_in_roi(box_xyxy, frame_shape, points):
    if not points:
        return True
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box_xyxy
    center = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height))
    return cv2.pointPolygonTest(np.asarray(points, dtype=np.float32), center, False) >= 0


def draw_roi(image, points):
    if not points:
        return image
    image = image.copy()
    height, width = image.shape[:2]
    pixels = np.rint(np.asarray(points) * [width - 1, height - 1]).astype(np.int32)
    cv2.polylines(image, [pixels], True, (0, 215, 255), 3)
    return image

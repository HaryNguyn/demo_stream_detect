import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from ultralytics.engine.results import Results

from vehicle_ai.config import PipelineConfig
from vehicle_ai.aggregation import Vehicle
from vehicle_ai.inference import draw_anpr_preview, draw_recognition
from vehicle_ai.roi import center_in_roi, validate_roi
from vehicle_ai.session import RecognitionSession


class ROITests(unittest.TestCase):
    def test_anpr_draws_plate_box_in_frame_coordinates(self):
        frame = np.full((480, 640, 3), 77, np.uint8)
        result = Results(frame, path="", names={0: "car"}, boxes=torch.tensor([
            [200, 200, 400, 400, 1, .9, 0],
        ]))
        vehicle = Vehicle(0, plate_box=(.25, .6, .75, .8),
                          plate_detection_score=.9, plate_ocr_score=.95)
        vehicle.add_plate("29A12345", .95)
        preview = draw_anpr_preview(frame, result, {1: vehicle}, inference_fps=12)
        # Plate at (250,320)-(350,360), rather than the entire vehicle.
        self.assertFalse(np.array_equal(preview[360, 300], frame[360, 300]))
        np.testing.assert_array_equal(preview[400, 300], frame[400, 300])
        self.assertEqual(preview.shape, frame.shape)
        self.assertTrue(np.all(frame == 77))
        outside = [[0, 0], [.2, 0], [.2, 1], [0, 1]]
        filtered = draw_anpr_preview(frame, result, {1: vehicle}, outside)
        np.testing.assert_array_equal(filtered[360, 300], frame[360, 300])

    def test_live_overlay_uses_current_video_frame(self):
        old = np.zeros((480, 640, 3), np.uint8)
        current = np.full_like(old, 77)
        result = Results(old, path="", names={0: "car"}, boxes=torch.tensor([
            [200, 200, 300, 300, 1, .9, 0],
        ]))
        roi = [[.25, .25], [.75, .25], [.75, .75], [.25, .75]]
        preview = draw_recognition(result, {}, roi, frame=current)
        np.testing.assert_array_equal(preview[0, 0], current[0, 0])
        self.assertFalse(np.array_equal(preview[300, 250], current[300, 250]))
        self.assertFalse(np.array_equal(preview[240, 160], current[240, 160]))
        self.assertTrue(np.all(current == 77))
        self.assertTrue(np.all(old == 0))

    def test_normalized_roi_works_for_multiple_resolutions(self):
        roi = validate_roi([[.25, .25], [.75, .25], [.75, .75], [.25, .75]])
        for scale in (1, 2, 4):
            shape = (100 * scale, 100 * scale, 3)
            self.assertTrue(center_in_roi([40*scale, 40*scale, 60*scale, 60*scale], shape, roi))
            self.assertFalse(center_in_roi([0, 0, 10*scale, 10*scale], shape, roi))
        self.assertTrue(center_in_roi([0, 0, 1, 1], (100, 100, 3), []))

    def test_invalid_polygons(self):
        for polygon in (
            [[0, 0], [1, 1]], [[0, 0], [.5, .5], [1, 1]],
            [[0, 0], [2, 0], [1, 1]], [[0, 0], [float("nan"), 0], [1, 1]],
            [[0, 0], [1, 1], [0, 1], [1, 0]],
        ):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                validate_roi(polygon)

    def test_session_ignores_vehicle_outside_roi(self):
        image = np.zeros((100, 100, 3), np.uint8)
        result = Results(image, path="", names={0: "car"}, boxes=torch.tensor([
            [40, 40, 60, 60, 1, .9, 0],
            [0, 0, 10, 10, 2, .9, 0],
        ]))
        detector = MagicMock()
        detector.track.return_value = [result]
        config = PipelineConfig(vehicle_weights="unused", vehicle_classes={0: "car"},
                                roi=[[.25, .25], [.75, .25], [.75, .75], [.25, .75]],
                                crop_vehicle_roi=False)
        session = RecognitionSession(detector, config, "cam")
        session.feed(image, 0)
        preview = draw_recognition(result, session.tracks, config.roi)
        self.assertEqual(preview.shape, image.shape)
        events = session.finalize()
        self.assertEqual([row["Track ID"] for row in events], [1])
        config.roi = []
        session = RecognitionSession(detector, config, "cam")
        session.feed(image, 0)
        self.assertEqual(len(session.finalize()), 2)


if __name__ == "__main__":
    unittest.main()

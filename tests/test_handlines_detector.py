import pytest
from modules import HandLinesDetector
import numpy as np
from PIL import Image

class TestHandLinesDetector:
    @pytest.fixture
    def detector(self):
        return HandLinesDetector()

    def test_initialization(self, detector):
        assert detector.angle_threshold == 15
        assert detector.distance_threshold == 25

    def test_merge_similar_lines(self):
        lines = [
            ((0, 0), (10, 10), 45),
            ((1, 1), (11, 11), 46),
            ((20, 20), (30, 30), 90)
        ]
        result = HandLinesDetector.merge_similar_lines(lines)
        assert len(result) > 0

    def test_process_image(self, detector):
        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        edges, lines = detector.process_image(test_image)
        assert isinstance(edges, np.ndarray)
        assert isinstance(lines, list)

    def test_call(self, detector):
        # Create test PIL Image
        test_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        result = detector(test_image)
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # RGB image
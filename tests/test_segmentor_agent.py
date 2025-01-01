import pytest
from modules import SegmentorAgent
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

class TestSegmentorAgent:
    @pytest.fixture
    def agent(self):
        return SegmentorAgent()

    @patch('lang_sam.LangSAM')
    def test_call(self, mock_langsam):
        agent = SegmentorAgent()
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image_bytes = Image.fromarray(test_image).tobytes()
        
        # Mock LangSAM predictions
        mock_mask = np.ones((100, 100), dtype=bool)
        mock_langsam.return_value.predict.return_value = [{'masks': mock_mask[None, None, ...]}]
        
        result = agent(test_image_bytes)
        assert isinstance(result, Image.Image)
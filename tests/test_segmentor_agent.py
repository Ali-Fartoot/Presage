# tests/test_segmentor_agent.py
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import io
from modules import SegmentorAgent
class TestSegmentorAgent:
    @pytest.fixture
    def agent(self):
        return SegmentorAgent()

    @patch('lang_sam.LangSAM')
    def test_call(self, mock_langsam):
        agent = SegmentorAgent()
        
        test_image = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        test_image_bytes = img_byte_arr.getvalue()
        
        mock_mask = np.ones((100, 100), dtype=bool)
        mock_langsam.return_value.predict.return_value = [{'masks': mock_mask[None, None, ...]}]
        
        result = agent(test_image_bytes)
        assert isinstance(result, Image.Image)
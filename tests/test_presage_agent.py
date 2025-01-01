import pytest
from modules import PresageAgent
from unittest.mock import MagicMock, patch
import base64

class TestPresageAgent:
    @pytest.fixture
    def agent(self):
        return PresageAgent()
    
    def convert_to_base64(self, image_bytes: bytes) -> str:
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

    def test_infer(self, agent):
        with open('../example/test.jpg', "rb") as image_file:
            test_base64 = self.convert_to_base64(image_file.read())
            result = agent.infer(test_base64)
            assert type(result) == str
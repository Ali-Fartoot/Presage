import pytest
from modules import FallbackAgent
from unittest.mock import MagicMock, patch
import base64

class TestFallbackAgent:
    @pytest.fixture
    def agent(self):
        return FallbackAgent()

    def test_initialization(self, agent):
        assert len(agent.message_template) == 2
        assert agent.message_template[0]["role"] == "system"
        assert agent.message_template[1]["role"] == "user"

    def convert_to_base64(self, image_bytes: bytes) -> str:
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"
    
    @patch('openai.OpenAI')
    def test_infer(self, agent):

        with open('../example/test.jpg', "rb") as image_file:
            test_base64 = self.convert_to_base64(image_file.read())
            result = agent.infer(test_base64)
            assert result.lower() == "yes" 

        with open('../example/flower.jpg', "rb") as image_file:
            test_base64 = self.convert_to_base64(image_file.read())
            result = agent.infer(test_base64)
            assert result.lower() == "no" 
import pytest
from modules import FallbackAgent
from unittest.mock import MagicMock, patch
import base64
from end_to_end_test import measure_process_time

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
    @measure_process_time
    def test_infer(self, agent):

        with open('./example/test.jpg', "rb") as image_file:
            test_base64 = self.convert_to_base64(image_file.read())
            result = agent.infer(test_base64)
            assert any(x in result.lower() for x in ["yes"])

        with open('./example/flower.jpg', "rb") as image_file:
            test_base64 = self.convert_to_base64(image_file.read())
            result = agent.infer(test_base64)
            assert any(x in result.lower() for x in ["no"])
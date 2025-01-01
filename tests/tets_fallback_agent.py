import pytest
from modules import FallbackAgent
from unittest.mock import MagicMock, patch

class TestFallbackAgent:
    @pytest.fixture
    def agent(self):
        return FallbackAgent()

    def test_initialization(self, agent):
        assert len(agent.message_template) == 2
        assert agent.message_template[0]["role"] == "system"
        assert agent.message_template[1]["role"] == "user"

    @patch('openai.OpenAI')
    def test_infer(self, mock_openai, agent):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Yes"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = agent.infer("test_base64")
        assert result == "Yes"
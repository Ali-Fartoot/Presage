import pytest
from modules import LLMAgent

class TestLLMAgent:
    class ConcreteAgent(LLMAgent):
        def infer(self, temperature=0.9, max_token=500, n=1, stop=None):
            return "test"

    def test_initialization(self):
        agent = self.ConcreteAgent()
        assert agent.message_template == []
        
    def test_custom_message_template(self):
        template = [{"role": "system", "content": "test"}]
        agent = self.ConcreteAgent(message=template)
        assert agent.message_template == template
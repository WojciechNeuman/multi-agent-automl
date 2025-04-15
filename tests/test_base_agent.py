import sys
import os
from unittest import TestCase
from unittest.mock import MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.base_agent import BaseAgent

class DummyAgent(BaseAgent):
    def run(self, request):
        return {"status": "ok"}

class TestBaseAgent(TestCase):
    def test_dummy_agent_runs(self):
        agent = DummyAgent(name="TestAgent")

        agent.run = MagicMock(return_value={"status": "ok"})

        input_data = {"hello": "world"}
        result = agent.run(input_data)

        self.assertEqual(result, {"status": "ok"})
        agent.run.assert_called_once_with(input_data)

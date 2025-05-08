import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from agents.base_agent import BaseAgent
from unittest import TestCase

class DummyAgent(BaseAgent):
    def run(self, request):
        return {"status": "ok"}

class TestBaseAgent(TestCase):
    def test_dummy_agent_runs(self):
        agent = DummyAgent(name="TestAgent")
        input_data = {"hello": "world"}
        self.assertEqual(agent.run(input_data), {"status": "ok"})

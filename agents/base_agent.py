from abc import ABC, abstractmethod
from typing import Any
from loguru import logger

class BaseAgent(ABC):
    """
    Abstract base class for all AI agents in the AutoML pipeline.
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        logger.info(f"[{self.name}] Initialized.")

    @abstractmethod
    def run(self, request: Any) -> Any:
        """
        The core method to be implemented by all concrete agents.
        """
        pass

    def receive(self, input_data: Any) -> Any:
        """
        Optional preprocessing step for input data.
        """
        logger.debug(f"[{self.name}] Received input: {type(input_data)}")
        return input_data

    def send(self, output: Any) -> Any:
        """
        Optional postprocessing step for output data.
        """
        logger.debug(f"[{self.name}] Sending output: {type(output)}")
        return output

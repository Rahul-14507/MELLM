from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("LLMRouter.Specialist")

class BaseSpecialist(ABC):
    """
    Abstract base class for all domain-specific specialists.
    Receives a Llama object from llama-cpp-python directly.
    """

    def __init__(self, model, tokenizer=None, max_new_tokens: int = 512):
        self.model = model
        # tokenizer is kept for API compatibility but is unused —
        # llama-cpp-python handles chat templating internally.
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a response for the given optimized prompt."""
        pass

    def _postprocess(self, output: str) -> str:
        """Cleans the output string."""
        return output.strip()

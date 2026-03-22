from abc import ABC, abstractmethod
from typing import Generator
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
        """Returns the full response as a string (non-streaming)."""
        pass

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        """
        Yields response tokens one by one as they are generated.
        Default implementation streams via create_chat_completion with stream=True.
        Subclasses can override if needed.
        """
        stream = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self._get_temperature(),
            stream=True
        )
        for chunk in stream:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

    def _get_temperature(self) -> float:
        """Override in subclasses to set domain-specific temperature."""
        return 0.7

    def _postprocess(self, output: str) -> str:
        """Cleans the output string."""
        return output.strip()

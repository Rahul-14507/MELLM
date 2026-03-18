from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("LLMRouter.Specialist")

class BaseSpecialist(ABC):
    """
    Abstract base class for all domain-specific specialists.
    """
    
    def __init__(self, model, tokenizer, max_new_tokens: int = 1024):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generates a response for the given optimized prompt.
        """
        pass

    def _postprocess(self, output: str) -> str:
        """
        Cleans the output string by removing EOS tokens and extra whitespace.
        """
        # Common cleanup
        cleaned = output.strip()
        # Some models might output tokens like <|endoftext|>
        stop_tokens = ["<|endoftext|>", "</s>", "<|im_end|>", "<|end|>"]
        for token in stop_tokens:
            cleaned = cleaned.replace(token, "")
        return cleaned.strip()

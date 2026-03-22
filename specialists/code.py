from .base_specialist import BaseSpecialist
from typing import Generator

class CodeSpecialist(BaseSpecialist):
    def _get_temperature(self) -> float:
        return 0.1  # deterministic for code

    def generate(self, prompt: str) -> str:
        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.1,
            stop=["```\n\n"]
        )
        return self._postprocess(
            response["choices"][0]["message"]["content"]
        )

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        stream = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.1,
            stream=True,
            stop=["```\n\n"]
        )
        for chunk in stream:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

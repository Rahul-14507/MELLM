from .base_specialist import BaseSpecialist
from typing import Generator

class MathSpecialist(BaseSpecialist):
    def _get_temperature(self) -> float:
        return 0.1

    def generate(self, prompt: str) -> str:
        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.1,
        )
        return self._postprocess(
            response["choices"][0]["message"]["content"]
        )

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        stream = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.1,
            stream=True
        )
        for chunk in stream:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token
